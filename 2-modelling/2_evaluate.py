"""
2_evaluate.py — PASO 2: evalúa el mejor modelo de una corrida sobre un split (por defecto
test) con las métricas a nivel de ventana y a nivel de video (puntuación MIL en dos etapas).

Lee la configuración y las clases de outputs/{run_id}/run.yaml y los pesos de best.pt;
imprime el reporte por consola y guarda outputs/{run_id}/metrics_{split}.json
(con las matrices de confusión). Con --wandb retoma la corrida de wandb y sube las métricas.

Uso
---
    python 2_evaluate.py --run-id wa4jsu91
    python 2_evaluate.py --run-id wa4jsu91 --split val
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 2-modelling/

import numpy as np
import torch

from lib.cli import add_override_args, add_tracking_args, parse_overrides, resolve_data_root
from lib.dataset import build_loaders
from lib.evaluation import evaluate_split, print_report
from lib.model import build_model
from lib.runs import OUTPUTS_DIR, load_run_info, run_dir
from lib.tracking import Tracker


def main():
    parser = argparse.ArgumentParser(description="Evalúa una corrida entrenada")
    parser.add_argument("--run-id", required=True, help="Id de la corrida (carpeta outputs/{id}/)")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR,
                        help="Carpeta donde están las corridas (por defecto: outputs/)")
    add_override_args(parser)
    add_tracking_args(parser)
    args = parser.parse_args()

    out_dir = run_dir(args.run_id, args.outputs_dir)
    cfg, classes, sources = load_run_info(out_dir)
    for key, value in parse_overrides(args).items():
        if not hasattr(cfg, key):
            raise SystemExit(f"campo desconocido en --set: {key}")
        setattr(cfg, key, tuple(value) if isinstance(value, list) else value)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = device == 'cuda'
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    normal_idx = cls_to_idx.get('Normal', -1)

    data_root = resolve_data_root(cfg.data_root)
    _, val_loader, test_loader, _, vl_recs, test_recs = build_loaders(
        data_root, sources, classes, cls_to_idx,
        cfg.seq_len, cfg.stride, tuple(cfg.img_size), cfg.batch_size, cfg.num_workers,
        max_windows_per_video=cfg.max_windows_per_video, frame_stride=cfg.frame_stride)
    loader, recs = (test_loader, test_recs) if args.split == 'test' else (val_loader, vl_recs)

    model = build_model(cfg, len(classes), pretrained=False).to(device)    # los pesos salen de best.pt
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device, weights_only=True))

    res = evaluate_split(model, loader, recs, classes, normal_idx, cfg, device, amp, split=args.split)
    print_report(res, classes, cfg)

    tracker = Tracker(out_dir, classes, use_wandb=args.wandb, project=cfg.wandb_project,
                      name=cfg.wandb_run_name, run_id=args.run_id)
    tracker.log(res['flat'], res['plots'])
    path = tracker.save_json(f'metrics_{args.split}.json', {
        'run_id': args.run_id, 'split': args.split, 'classes': classes,
        'metrics': res['flat'],
        'confusion_matrix_window': res['window']['metrics']['confusion_matrix'],
        'confusion_matrix_video': res['video']['metrics']['confusion_matrix'],
    })
    tracker.finish()
    print(f'Métricas guardadas en {path}')


if __name__ == "__main__":
    main()
