"""
ablation/suite.py — define, entrena y evalúa las corridas de una ablación.

Una *suite* es un YAML (config/*.yaml) con una configuración base y una lista de variantes.
Cada variante cambia una sola cosa respecto de `reference`:

    - name: temporal_mean              # id de la variante
      model: {temporal: mean}          # variante de arquitectura (ver variants.py)
      set: {label_smoothing: 0.0}      # campos de Config de la etapa 2 que se pisan
      sampler: uniform                 # 'balanced' (por defecto, frecuencia inversa) o 'uniform'

Cada (variante, semilla) es una corrida de la etapa 2: outputs/{suite}/{variante}_s{semilla}/
con best.pt, last.pt, run.yaml, history.csv y además, escritos aquí,
  metrics_val.json, metrics_test.json   mismas métricas que 2_evaluate.py
  preds_val.npz, preds_test.npz          probabilidades por ventana (para re-puntuar sin GPU)
  variant.json                           qué se cambió, parámetros, época del mejor modelo
Una corrida con metrics_test.json se da por terminada; con best.pt pero sin métricas solo se
evalúa; con last.pt se reanuda (el comportamiento de lib/pipeline.py).
"""
import gc
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from lib.config import Config
from lib.evaluation import score_predictions
from lib.model import count_parameters
from lib.pipeline import train_run
from lib.runs import load_run_info
from lib.training import evaluate

from .variants import build_ablation_model


@dataclass
class RunSpec:
    variant: str
    seed: int
    overrides: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    sampler: str = 'balanced'

    @property
    def run_id(self) -> str:
        return f'{self.variant}_s{self.seed}'


# ── Definición de la suite ─────────────────────────────────────────────────────

def load_suite(path) -> dict:
    path = Path(path)
    suite = yaml.safe_load(path.read_text())
    suite['_dir'] = path.parent
    suite.setdefault('name', path.stem)
    suite.setdefault('reference', 'reference')
    suite.setdefault('common', {})
    suite.setdefault('seeds', [42])
    names = [v['name'] for v in suite['variants']]
    if len(set(names)) != len(names):
        raise SystemExit('hay variantes con el mismo nombre en la suite')
    if suite['reference'] not in names:
        raise SystemExit(f"la suite debe incluir la variante de referencia {suite['reference']!r}")
    for v in suite['variants']:
        if v.get('sampler', 'balanced') not in ('balanced', 'uniform'):
            raise SystemExit(f"{v['name']}: sampler debe ser 'balanced' o 'uniform'")
    return suite


def expand(suite, only=None, seeds=None) -> list[RunSpec]:
    seeds = seeds or suite['seeds']
    specs = []
    for v in suite['variants']:
        if only and v['name'] not in only and v['name'] != suite['reference']:
            continue
        for seed in seeds:
            specs.append(RunSpec(v['name'], int(seed), dict(v.get('set') or {}),
                                 dict(v.get('model') or {}), v.get('sampler', 'balanced')))
    if only:
        unknown = set(only) - {v['name'] for v in suite['variants']}
        if unknown:
            raise SystemExit(f'variantes desconocidas en --only: {sorted(unknown)}')
    return specs


def make_config(suite, spec: RunSpec, cli_overrides=None) -> Config:
    base = (suite['_dir'] / suite['base_config']).resolve()
    overrides = {**suite['common'], **spec.overrides, **(cli_overrides or {}),
                 'seed': spec.seed, 'wandb_run_name': f"{suite['name']}-{spec.run_id}"}
    return Config.from_yaml(base, **overrides)


# ── Una corrida ───────────────────────────────────────────────────────────────

def _uniform_train_loader(loader, cfg):
    """Mismo dataset de entrenamiento, pero con orden aleatorio uniforme (sin balancear por clase)."""
    return DataLoader(loader.dataset, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers, pin_memory=True)


def _json_default(o):
    return o.tolist() if hasattr(o, 'tolist') else float(o)


def evaluate_run(out_dir: Path, loaders: dict, factory, spec: RunSpec | None = None):
    """Evalúa best.pt en val y test; guarda métricas, predicciones por ventana y variant.json."""
    out_dir = Path(out_dir)
    cfg, classes, _ = load_run_info(out_dir)
    normal_idx = classes.index('Normal') if 'Normal' in classes else -1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = device == 'cuda'
    model = factory(cfg, len(classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device, weights_only=True))

    for split in ('val', 'test'):
        loader, recs = loaders[f'{split}_loader'], loaders[f'{split}_recs']
        probs, preds, labels = evaluate(model, loader, device, len(classes), amp=amp)
        res = score_predictions(probs, preds, labels, recs, classes, normal_idx, cfg, split)
        np.savez_compressed(out_dir / f'preds_{split}.npz', probs=probs, preds=preds, labels=labels,
                            vpaths=np.array([r[0] for r in recs]), starts=np.array([r[1] for r in recs]))
        (out_dir / f'metrics_{split}.json').write_text(json.dumps({
            'run_id': out_dir.name, 'split': split, 'classes': classes, 'metrics': res['flat'],
            'confusion_matrix_window': res['window']['metrics']['confusion_matrix'],
            'confusion_matrix_video': res['video']['metrics']['confusion_matrix'],
        }, indent=2, ensure_ascii=False, default=_json_default))

    total, trainable, _ = count_parameters(model)
    info = {'run_id': out_dir.name, 'params_total': total, 'params_trainable': trainable}
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
        info.update({'variant': spec.variant, 'seed': spec.seed, 'set': spec.overrides,
                     'model': spec.model, 'sampler': spec.sampler})
    (out_dir / 'variant.json').write_text(json.dumps(info, indent=2, ensure_ascii=False))
    del model


def run_one(suite, spec: RunSpec, outputs_dir: Path, cli_overrides=None, use_wandb=False, retrain=False):
    out_dir = outputs_dir / spec.run_id
    if (out_dir / 'metrics_test.json').exists() and not retrain:
        print(f'[{spec.run_id}] ya terminada (metrics_test.json) — se omite (usa --retrain para repetirla)')
        return out_dir
    cfg = make_config(suite, spec, cli_overrides)
    held: dict = {}

    def loaders_hook(loaders):
        tl, vl, tel, trr, vlr, ter = loaders
        if spec.sampler == 'uniform':
            tl = _uniform_train_loader(tl, cfg)
        held.update(val_loader=vl, test_loader=tel, val_recs=vlr, test_recs=ter)
        return tl, vl, tel, trr, vlr, ter

    def factory(c, n_classes, pretrained=None):
        return build_ablation_model(c, n_classes, pretrained, **spec.model)

    train_run(cfg, spec.run_id, outputs_dir, use_wandb=use_wandb, retrain=retrain,
              model_factory=factory, loaders_hook=loaders_hook,
              extra_info={'ablation': {'suite': suite['name'], 'variant': spec.variant,
                                       'model': spec.model, 'sampler': spec.sampler, 'set': spec.overrides}})
    evaluate_run(out_dir, held, factory, spec)
    return out_dir


def run_suite(suite, specs, outputs_dir: Path, cli_overrides=None, use_wandb=False, retrain=False):
    """Corre todas las corridas; una que falle no detiene a las demás. Devuelve las que fallaron."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, spec in enumerate(specs, 1):
        print(f'\n{"=" * 78}\n[{i}/{len(specs)}] {spec.run_id}   (set={spec.overrides} model={spec.model} sampler={spec.sampler})\n{"=" * 78}')
        try:
            run_one(suite, spec, outputs_dir, cli_overrides, use_wandb, retrain)
        except Exception:
            traceback.print_exc()
            failed.append(spec.run_id)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return failed
