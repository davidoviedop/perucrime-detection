"""
ablation/suite.py — define, entrena y evalúa las corridas de una ablación.

Una *suite* es un YAML (config/*.yaml) con una configuración base y una lista de variantes; cada
variante cambia un campo de Config de la etapa 2 respecto de la referencia (diseño OFAT: un factor a
la vez), y se corre con varias semillas:

    - id: A1
      question: ¿Hace falta modelar el tiempo?
      set: {temporal_module: none}

Cada (variante, semilla) es una corrida de la etapa 2: outputs/{suite}/{id}_s{semilla}/ con best.pt,
last.pt, run.yaml, history.csv y, escritos aquí:
  metrics_val.json, metrics_test.json   mismas métricas que 2_evaluate.py
  preds_val.npz, preds_test.npz          probabilidades por ventana (para M1–M12 y el bootstrap)
  variant.json                           qué se cambió, parámetros, GFLOPs, época del mejor modelo, tiempo
Una corrida con las métricas escritas se da por terminada; con last.pt se reanuda; con best.pt pero sin
métricas solo se evalúa (el comportamiento de lib/pipeline.py). Reglas de la suite (ver README): todas
las corridas usan el mismo split, el mismo protocolo y las mismas semillas.
"""
import gc
import json
import platform
import shlex
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from lib.cli import resolve_data_root
from lib.config import Config
from lib.dataset import build_loaders
from lib.evaluation import score_predictions, save_predictions
from lib.model import build_model, count_parameters
from lib.pipeline import train_run
from lib.runs import load_run_info
from lib.training import evaluate
from run_report import RunReport, current

from . import STAGE_DIR

@dataclass
class RunSpec:
    vid: str                   # id de la variante: A0, A1, …
    seed: int
    overrides: dict = field(default_factory=dict)
    question: str = ''

    @property
    def run_id(self) -> str:
        return f'{self.vid}_s{self.seed}'


# ── Definición de la suite ─────────────────────────────────────────────────────

def load_suite(path) -> dict:
    path = Path(path)
    suite = yaml.safe_load(path.read_text())
    suite['_dir'] = path.parent
    suite['_path'] = path.resolve()
    suite.setdefault('name', path.stem)
    suite.setdefault('common', {})
    suite.setdefault('seeds', [42, 43, 44])
    suite.setdefault('eval_splits', ['val', 'test'])
    ids = [v['id'] for v in suite['variants']]
    if len(set(ids)) != len(ids):
        raise SystemExit('hay variantes con el mismo id en la suite')
    suite.setdefault('reference', ids[0])
    if suite['reference'] not in ids:
        raise SystemExit(f"la suite debe incluir la variante de referencia {suite['reference']!r}")
    return suite


def expand(suite, only=None, seeds=None) -> list[RunSpec]:
    only = set(only or [])
    unknown = only - {v['id'] for v in suite['variants']}
    if unknown:
        raise SystemExit(f'variantes desconocidas en --only: {sorted(unknown)}')
    specs = []
    for v in suite['variants']:
        if only and v['id'] not in only and v['id'] != suite['reference']:
            continue
        for seed in (seeds or suite['seeds']):
            specs.append(RunSpec(v['id'], int(seed), dict(v.get('set') or {}), v.get('question', '')))
    return specs


def make_config(suite, spec: RunSpec, cli_overrides=None) -> Config:
    base = (suite['_dir'] / suite['base_config']).resolve()
    overrides = {**suite['common'], **spec.overrides, **(cli_overrides or {}),
                 'seed': spec.seed, 'wandb_run_name': f'abl-{spec.vid}-s{spec.seed}'}
    return Config.from_yaml(base, **overrides)


def environment_info() -> dict:
    """Se guarda en run.yaml (`extra`) para saber en qué servidor y con qué versiones corrió cada corrida."""
    return {'server': platform.node(), 'torch': str(torch.__version__), 'timm': str(timm.__version__),
            'cuda': str(torch.version.cuda), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


# ── Una corrida ───────────────────────────────────────────────────────────────

def _json_default(o):
    return o.tolist() if hasattr(o, 'tolist') else float(o)


def gflops_per_window(model, cfg, device) -> float:
    """GFLOPs (2 × MACs) de una pasada de una ventana (1, T, 3, H, W) con FlopCounterMode.
    OJO: FlopCounterMode no cuenta las celdas LSTM (A2); su aporte es despreciable frente al backbone."""
    model.eval()
    x = torch.zeros(1, cfg.seq_len, 3, cfg.img_size[0], cfg.img_size[1], device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops() / 1e9


def evaluate_run(out_dir: Path, splits, spec: RunSpec | None = None, train_seconds=None):
    """Evalúa best.pt en los splits indicados; guarda métricas, predicciones por ventana y variant.json."""
    out_dir = Path(out_dir)
    cfg, classes, sources = load_run_info(out_dir)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    normal_idx = cls_to_idx.get('Normal', -1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = device == 'cuda'

    _, val_loader, test_loader, _, vl_recs, test_recs = build_loaders(
        resolve_data_root(cfg.data_root), sources, classes, cls_to_idx, cfg.seq_len, cfg.stride,
        tuple(cfg.img_size), cfg.batch_size, cfg.num_workers,
        max_windows_per_video=cfg.max_windows_per_video, frame_stride=cfg.frame_stride)
    loaders = {'val': (val_loader, vl_recs), 'test': (test_loader, test_recs)}

    model = build_model(cfg, len(classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device, weights_only=True))

    for split in splits:
        loader, recs = loaders[split]
        probs, preds, labels = evaluate(model, loader, device, len(classes), amp=amp)
        res = score_predictions(probs, preds, labels, recs, classes, normal_idx, cfg, split)
        save_predictions(out_dir / f'preds_{split}.npz', probs, preds, labels, recs)
        (out_dir / f'metrics_{split}.json').write_text(json.dumps({
            'run_id': out_dir.name, 'split': split, 'classes': classes, 'metrics': res['flat'],
            'confusion_matrix_window': res['window']['metrics']['confusion_matrix'],
            'confusion_matrix_video': res['video']['metrics']['confusion_matrix'],
        }, indent=2, ensure_ascii=False, default=_json_default))

    total, trainable, _ = count_parameters(model)
    info = {'run_id': out_dir.name, 'params_total': total, 'params_trainable': trainable,
            'gflops_window': gflops_per_window(model, cfg, device)}
    if train_seconds is not None:
        info['train_seconds'] = round(train_seconds, 1)
    hist = out_dir / 'history.csv'
    if hist.exists():
        import csv
        rows = list(csv.DictReader(hist.open(newline='', encoding='utf-8')))
        if rows:
            info['epochs_run'] = len(rows)
            best = max(rows, key=lambda r: float(r['val_auc']))
            info['best_epoch'] = int(float(best['epoch']))
            info['best_val_auc'] = float(best['val_auc'])
    if spec is not None:
        info.update({'id': spec.vid, 'seed': spec.seed, 'set': spec.overrides, 'question': spec.question})
    info.update(environment_info())
    (out_dir / 'variant.json').write_text(json.dumps(info, indent=2, ensure_ascii=False))
    del model


def run_command(suite, spec: RunSpec, outputs_dir: Path, cli_overrides, use_wandb, retrain) -> str:
    """Comando que repite exactamente esta corrida (una variante y una semilla)."""
    try:
        suite_arg = suite['_path'].relative_to(STAGE_DIR).as_posix()
    except ValueError:
        suite_arg = str(suite['_path'])
    parts = ['python', '1_run_ablation.py', '--suite', suite_arg, '--only', spec.vid, '--seeds', str(spec.seed),
             '--outputs-dir', str(outputs_dir)]
    for k, v in (cli_overrides or {}).items():
        parts += ['--set', f'{k}={v if isinstance(v, str) else json.dumps(v)}']
    if use_wandb:
        parts.append('--wandb')
    if retrain:
        parts.append('--retrain')
    return shlex.join(parts)


def run_one(suite, spec: RunSpec, outputs_dir: Path, cli_overrides=None, use_wandb=False, retrain=False):
    out_dir = outputs_dir / spec.run_id
    done = all((out_dir / f'metrics_{s}.json').exists() for s in suite['eval_splits'])
    if done and not retrain:
        print(f'[{spec.run_id}] ya terminada — se omite (usa --retrain para repetirla)')
        return out_dir
    cfg = make_config(suite, spec, cli_overrides)
    # Un reporte por corrida (variante × semilla) en 5-ablation/reports/, con todos los parámetros y métricas.
    with RunReport(STAGE_DIR, 'ablation_run', run_id=spec.run_id, command=run_command(suite, spec, outputs_dir, cli_overrides,
                                                                                       use_wandb, retrain)) as report:
        report.add_info({'suite': suite['name'], 'variant': spec.vid, 'seed': spec.seed, 'question': spec.question,
                         'changes_vs_reference': spec.overrides, 'cli_overrides': cli_overrides or {},
                         'eval_splits': suite['eval_splits'], 'outputs_dir': str(outputs_dir)})
        report.add_file('suite', suite['_path'])
        report.add_file('base_config', (suite['_dir'] / suite['base_config']).resolve())
        t0 = time.time()
        train_run(cfg, spec.run_id, outputs_dir, use_wandb=use_wandb, retrain=retrain,
                  extra_info={'ablation': {'suite': suite['name'], 'id': spec.vid, 'question': spec.question,
                                           'set': spec.overrides, **environment_info()}})
        evaluate_run(out_dir, suite['eval_splits'], spec, train_seconds=time.time() - t0)
        # Todas las métricas quedan también en el reporte (además de metrics_*.json), para conservarlas entre corridas.
        variant = json.loads((out_dir / 'variant.json').read_text())
        report.add_results({k: v for k, v in variant.items() if k in ('params_total', 'params_trainable', 'gflops_window',
                                                                     'train_seconds', 'epochs_run', 'best_epoch', 'best_val_auc')})
        for split in suite['eval_splits']:
            report.add_results(json.loads((out_dir / f'metrics_{split}.json').read_text())['metrics'], section=f'metrics_{split}')
        for name in ('best.pt', 'last.pt', 'run.yaml', 'history.csv', 'variant.json', 'metrics_val.json', 'metrics_test.json',
                     'preds_val.npz', 'preds_test.npz'):
            if (out_dir / name).exists():
                report.add_output(out_dir / name)
    return out_dir


def run_suite(suite, specs, outputs_dir: Path, cli_overrides=None, use_wandb=False, retrain=False):
    """Corre todas las corridas; una que falle no detiene a las demás. Devuelve las que fallaron."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    skipped = []
    for i, spec in enumerate(specs, 1):
        print(f'\n{"=" * 78}\n[{i}/{len(specs)}] {spec.run_id}  {spec.question}  set={spec.overrides}\n{"=" * 78}')
        try:
            if all((outputs_dir / spec.run_id / f'metrics_{s}.json').exists() for s in suite['eval_splits']) and not retrain:
                skipped.append(spec.run_id)
            run_one(suite, spec, outputs_dir, cli_overrides, use_wandb, retrain)
        except Exception:
            traceback.print_exc()
            failed.append(spec.run_id)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    current().add_results({'runs_requested': [s.run_id for s in specs], 'runs_failed': failed, 'runs_skipped_already_done': skipped,
                           'runs_executed': [s.run_id for s in specs if s.run_id not in failed and s.run_id not in skipped]})
    current().add_output(outputs_dir)
    return failed
