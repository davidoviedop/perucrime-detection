"""
1_train.py — PASO 1: entrena el detector de anomalías (backbone por fotograma +
transformer temporal) sobre las ventanas del dataset.

Una corrida = una carpeta outputs/{run_id}/ con
  best.pt, last.pt   mejor modelo (según val_auc por ventana) y último checkpoint
  run.yaml           configuración y clases con las que se entrenó
  history.csv        métricas por época (ventana y video MIL en validación)

El entrenamiento es a nivel de ventana. La puntuación MIL por video (lib/mil.py) solo
cambia cómo se reportan las métricas de validación, no el entrenamiento; ver README.md.

Si la carpeta de la corrida ya tiene best.pt se da por entrenada y no se entrena de nuevo
(usa --retrain para forzarlo); si solo tiene last.pt, se reanuda desde ahí.

Uso
---
    python 1_train.py                                   # profesor ConvNeXtV2-base (config/teacher_convnextv2_base.yaml)
    python 1_train.py --set epochs=40 --wandb           # entrenamiento largo con wandb
    python 1_train.py --config config/mi_config.yaml --run-id abc12345
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 2-modelling/

import numpy as np
import torch
import torch.nn as nn

from lib.cli import STAGE_DIR, add_override_args, add_tracking_args, parse_overrides, resolve_data_root
from lib.config import Config
from lib.dataset import build_loaders, discover_classes, list_sources
from lib.model import build_model, count_parameters
from lib.runs import OUTPUTS_DIR, new_run_id, run_dir, save_run_info
from lib.tracking import Tracker
from lib.training import build_optimizer, build_scheduler, fit, restore_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Entrena el detector de anomalías")
    parser.add_argument("--config", type=Path, default=STAGE_DIR / "config" / "teacher_convnextv2_base.yaml",
                        help="YAML con los hiperparámetros (por defecto: el profesor ConvNeXtV2-base)")
    parser.add_argument("--run-id", default=None,
                        help="Id de la corrida (por defecto se genera uno). Si ya existe outputs/{id}/ se reutiliza o reanuda")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR,
                        help="Carpeta donde se guardan las corridas (por defecto: outputs/)")
    parser.add_argument("--retrain", action="store_true",
                        help="Entrena aunque la corrida ya tenga best.pt")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="No descarga los pesos preentrenados del backbone (útil para pruebas)")
    add_override_args(parser)
    add_tracking_args(parser)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config, **parse_overrides(args))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = device == 'cuda'   # autocast + GradScaler: reduce a la mitad la memoria de activaciones en CUDA
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    run_id = args.run_id or new_run_id()
    out_dir = run_dir(run_id, args.outputs_dir)
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

    train_loader, val_loader, test_loader, tr_recs, vl_recs, test_recs = build_loaders(
        data_root, sources, classes, cls_to_idx,
        cfg.seq_len, cfg.stride, tuple(cfg.img_size), cfg.batch_size, cfg.num_workers,
        max_windows_per_video=cfg.max_windows_per_video, frame_stride=cfg.frame_stride)

    # ── Modelo ──────────────────────────────────────────────────────────────
    model = build_model(cfg, len(classes), pretrained=False if args.no_pretrained else None).to(device)
    total, trainable, frozen = count_parameters(model)
    print(f'Total parameters     : {total:,}')
    print(f'Trainable parameters : {trainable:,}')
    print(f'Frozen parameters    : {frozen:,}  (backbone early layers)')

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    print(f'Loss: CrossEntropyLoss  label_smoothing={cfg.label_smoothing}  (unweighted — sampler handles balance)')
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = torch.amp.GradScaler('cuda', enabled=amp)

    save_run_info(out_dir, cfg, classes, sources, run_id)

    # ── ¿Ya entrenado? ¿Reanudar? ───────────────────────────────────────────
    already_trained = best_path.exists() and not args.retrain
    resume = None
    if already_trained:
        print(f'{best_path} exists — loading it and skipping training (use --retrain to train again).')
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    elif last_path.exists():
        resume = restore_checkpoint(last_path, model, optimizer, scheduler, scaler, amp, device)
    else:
        print('No checkpoint found — starting from scratch.')

    tracker = Tracker(out_dir, classes, use_wandb=args.wandb, project=cfg.wandb_project,
                      name=cfg.wandb_run_name, run_id=run_id,
                      config={**cfg.to_dict(), 'n_classes': len(classes), 'classes': classes, 'sources': sources,
                              'amp': amp, 'video_scoring': 'mil', 'train_windows': len(tr_recs),
                              'val_windows': len(vl_recs), 'test_windows': len(test_recs)})
    if tracker.url:
        print(f'W&B run: {tracker.url}')

    # ── Entrenamiento ───────────────────────────────────────────────────────
    if not already_trained:
        if resume is not None and resume['start_epoch'] > cfg.epochs:
            print(f'Training already complete ({cfg.epochs} epochs). Loading best model.')
            best_val_auc, best_state = resume['best_val_auc'], resume['best_state']
        else:
            best_val_auc, best_state, _ = fit(
                model, train_loader, val_loader, vl_recs, criterion, optimizer, scheduler, scaler,
                cfg, classes, normal_idx, device, amp, tracker, last_path, run_id, resume=resume)
        if best_state:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), best_path)
        print(f'\nBest val AUC: {best_val_auc:.4f}')
        print(f'Saved best model to {best_path}')
        tracker.log_artifact(last_path, f'{cfg.wandb_run_name or run_id}-checkpoint', 'checkpoint')
        tracker.log_artifact(best_path, f'{cfg.wandb_run_name or run_id}-best-model', 'model')

    tracker.finish()
    print(f'\nSiguiente paso: python 2_evaluate.py --run-id {run_id}')


if __name__ == "__main__":
    main()
