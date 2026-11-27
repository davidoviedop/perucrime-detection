"""
lib/pipeline.py — flujo completo de un entrenamiento: datos → modelo → fit → guardar.

Lo usan 1_train.py (profesor) y, con otros ganchos, la destilación de la etapa 3. Los
ganchos permiten cambiar solo lo que difiere sin duplicar el flujo:

  criterion         pérdida a usar (por defecto, entropía cruzada con label smoothing)
  wrap_train_model  función (modelo, dispositivo) → módulo que se usa en la época de
                    entrenamiento (la destilación devuelve un envoltorio que además corre
                    al profesor); se llama solo si realmente se va a entrenar
  extra_info        dict que se guarda en la sección `extra` de run.yaml
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:                                    # reporte del experimento (workspace/run_report.py); sin él, no hace nada
    from run_report import current
except ImportError:                     # pragma: no cover
    class _NoReport:
        def __getattr__(self, name):
            return lambda *a, **k: None
    current = lambda: _NoReport()

from .cli import resolve_data_root
from .config import Config
from .dataset import build_loaders, discover_classes, list_sources
from .model import build_model, count_parameters
from .runs import run_dir, save_run_info
from .tracking import Tracker
from .training import build_optimizer, build_scheduler, fit, restore_checkpoint


def train_run(cfg: Config, run_id: str, outputs_dir: Path, use_wandb: bool = False,
              retrain: bool = False, pretrained=None, criterion=None, wrap_train_model=None,
              extra_info: dict | None = None, extra_config: dict | None = None) -> Path:
    """Entrena (o reanuda, o da por entrenada) la corrida `run_id` y devuelve su carpeta."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    report = current()
    report.set_run_id(run_id)
    report.set_config('config', cfg.to_dict())
    if extra_info:
        report.set_config('extra_info', extra_info)
    if extra_config:
        report.set_config('extra_config', extra_config)
    report.add_info({'retrain': retrain, 'pretrained_override': pretrained})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = device == 'cuda'   # autocast + GradScaler: reduce a la mitad la memoria de activaciones en CUDA
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    out_dir = run_dir(run_id, outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / 'best.pt', out_dir / 'last.pt'
    print(f'Run ID  : {run_id}')
    print(f'Outputs : {out_dir}')

    # ── Datos ───────────────────────────────────────────────────────────────
    data_root = resolve_data_root(cfg.data_root)
    sources    = list_sources(data_root)
    classes    = discover_classes(data_root, sources, split='train')
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    normal_idx = cls_to_idx.get('Normal', -1)
    print(f'Device  : {device}')
    print(f'Data    : {data_root}')
    print(f'Sources : {sources}')
    print(f'Classes : {len(classes)}  {classes}')
    print(f'Input   : ({cfg.seq_len}, 3, {cfg.img_size[0]}, {cfg.img_size[1]})')
    print(f'Normal class index: {normal_idx}')
    report.add_input('data_root', data_root)
    report.add_info({'device': device, 'amp': amp, 'classes': classes, 'sources': sources})

    train_loader, val_loader, test_loader, tr_recs, vl_recs, test_recs = build_loaders(
        data_root, sources, classes, cls_to_idx,
        cfg.seq_len, cfg.stride, tuple(cfg.img_size), cfg.batch_size, cfg.num_workers,
        max_windows_per_video=cfg.max_windows_per_video, frame_stride=cfg.frame_stride,
        temporal_jitter=cfg.temporal_jitter)
    report.add_info({'train_windows': len(tr_recs), 'val_windows': len(vl_recs), 'test_windows': len(test_recs)})
    # ── Modelo ──────────────────────────────────────────────────────────────
    model = build_model(cfg, len(classes), pretrained=pretrained).to(device)
    total, trainable, frozen = count_parameters(model)
    print(f'Total parameters     : {total:,}')
    print(f'Trainable parameters : {trainable:,}')
    print(f'Frozen parameters    : {frozen:,}  (backbone early layers)')
    report.add_info({'params_total': total, 'params_trainable': trainable, 'params_frozen': frozen})
    report.add_output(out_dir, 'carpeta de la corrida (best.pt, last.pt, run.yaml, history.csv)')
    report.flush()

    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        print(f'Loss: CrossEntropyLoss  label_smoothing={cfg.label_smoothing}  (unweighted — sampler handles balance)')
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = torch.amp.GradScaler('cuda', enabled=amp)

    save_run_info(out_dir, cfg, classes, sources, run_id, extra=extra_info)

    # ── ¿Ya entrenado? ¿Reanudar? ───────────────────────────────────────────
    already_trained = best_path.exists() and not retrain
    resume = None
    if already_trained:
        print(f'{best_path} exists — loading it and skipping training (use --retrain to train again).')
        report.note(f'{best_path.name} ya existía: no se entrenó (se cargó el mejor modelo). Usa --retrain para repetir.')
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    elif last_path.exists():
        resume = restore_checkpoint(last_path, model, optimizer, scheduler, scaler, amp, device)
        report.note(f'Se reanudó desde {last_path.name}' + (f' (época {resume["start_epoch"] - 1})' if resume else ' (checkpoint incompatible: se empezó de cero)'))
    else:
        print('No checkpoint found — starting from scratch.')

    tracker = Tracker(out_dir, classes, use_wandb=use_wandb, project=cfg.wandb_project,
                      name=cfg.wandb_run_name, run_id=run_id,
                      config={**cfg.to_dict(), **(extra_config or {}), 'n_classes': len(classes),
                              'classes': classes, 'sources': sources, 'amp': amp, 'video_scoring': 'mil',
                              'train_windows': len(tr_recs), 'val_windows': len(vl_recs),
                              'test_windows': len(test_recs)})
    if tracker.url:
        print(f'W&B run: {tracker.url}')

    # ── Entrenamiento ───────────────────────────────────────────────────────
    if not already_trained:
        history = None
        if resume is not None and resume['start_epoch'] > cfg.epochs:
            print(f'Training already complete ({cfg.epochs} epochs). Loading best model.')
            best_val_auc, best_state = resume['best_val_auc'], resume['best_state']
            history = resume['history']
        else:
            train_model = wrap_train_model(model, device) if wrap_train_model else None
            best_val_auc, best_state, history = fit(
                model, train_loader, val_loader, vl_recs, criterion, optimizer, scheduler, scaler,
                cfg, classes, normal_idx, device, amp, tracker, last_path, run_id,
                resume=resume, train_model=train_model)
        if best_state:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), best_path)
        print(f'\nBest val AUC: {best_val_auc:.4f}')
        print(f'Saved best model to {best_path}')
        results = {'best_val_auc': best_val_auc}
        if history and history.get('val_auc'):
            va = history['val_auc']
            results.update({'epochs_trained': len(va), 'epochs_configured': cfg.epochs,
                            'best_epoch': int(max(range(len(va)), key=lambda i: va[i]) + 1),
                            'last_epoch_train_auc': history['train_auc'][-1], 'last_epoch_val_auc': va[-1],
                            'stopped_early': len(va) < cfg.epochs})
        if device == 'cuda':
            results['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        report.add_results(results)
        for name in ('best.pt', 'last.pt', 'history.csv', 'run.yaml'):
            if (out_dir / name).exists():
                report.add_output(out_dir / name)
        tracker.log_artifact(last_path, f'{cfg.wandb_run_name or run_id}-checkpoint', 'checkpoint')
        tracker.log_artifact(best_path, f'{cfg.wandb_run_name or run_id}-best-model', 'model')

    tracker.finish()
    return out_dir
