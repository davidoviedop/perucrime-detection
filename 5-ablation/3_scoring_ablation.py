"""
3_scoring_ablation.py — PASO 3 (sin entrenar): ablación de la puntuación por VIDEO.

Toma una corrida ya entrenada (normalmente la referencia) y re-puntúa sus videos de test (y
val) con otras formas de agrupar las ventanas, a partir de las probabilidades guardadas
(preds_{split}.npz): distintos k y umbrales de la puntuación MIL de lib/mil.py, el voto por
mayoría de todas las ventanas, la probabilidad media y el max-pooling. Es rápido y no usa GPU.

Con --shuffle-frames evalúa además el mismo modelo con el orden de los fotogramas de cada
ventana barajado: mide cuánto usa realmente el modelo el orden temporal (necesita GPU y los datos).

Uso
---
    python 3_scoring_ablation.py --run-dir outputs/mobilevit_xs/reference_s42
    python 3_scoring_ablation.py --run-dir outputs/mobilevit_xs/reference_s42 --shuffle-frames
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 5-ablation/

import numpy as np

from ablation import RESULTS_DIR
from lib.runs import load_run_info, load_run_extra
from ablation.mil_variants import video_metrics

TOP_K = [0.05, 0.10, 0.20, 0.50, 1.00]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
KEYS = [('video_auc_macro', 'AUC macro'), ('video_f1_macro', 'F1 macro'), ('video_auc_binary', 'AUC binario'),
        ('video_accuracy', 'Exactitud')]


def load_preds(run_dir, split):
    z = np.load(run_dir / f'preds_{split}.npz')
    recs = [(str(v), int(s), int(l)) for v, s, l in zip(z['vpaths'], z['starts'], z['labels'])]
    return z['probs'], z['preds'], z['labels'], recs


def shuffled_predictions(run_dir, cfg, classes, sources, split, seed=0):
    """Predicciones del mejor modelo con el orden de los fotogramas de cada ventana barajado."""
    import torch
    from ablation.variants import build_ablation_model
    from lib.cli import resolve_data_root
    from lib.dataset import build_loaders
    from lib.training import evaluate

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    _, vl, tl, _, vr, tr = build_loaders(
        resolve_data_root(cfg.data_root), sources, classes, cls_to_idx, cfg.seq_len, cfg.stride,
        tuple(cfg.img_size), cfg.batch_size, cfg.num_workers,
        max_windows_per_video=cfg.max_windows_per_video, frame_stride=cfg.frame_stride)
    loader, recs = (tl, tr) if split == 'test' else (vl, vr)
    opts = (load_run_extra(run_dir).get('ablation') or {}).get('model') or {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_ablation_model(cfg, len(classes), pretrained=False, **opts).to(device)
    model.load_state_dict(torch.load(run_dir / 'best.pt', map_location=device, weights_only=True))

    g = torch.Generator().manual_seed(seed)

    class Shuffled(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x[:, torch.randperm(x.shape[1], generator=g).to(x.device)])

    probs, preds, labels = evaluate(Shuffled(model), loader, device, len(classes), amp=device == 'cuda')
    return probs, preds, labels, recs


def main():
    parser = argparse.ArgumentParser(description="Ablación de la puntuación por video")
    parser.add_argument("--run-dir", type=Path, required=True, help="Carpeta de la corrida (outputs/.../reference_s42)")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--shuffle-frames", action="store_true",
                        help="Evalúa también con el orden de los fotogramas barajado (GPU + datos)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cfg, classes, sources = load_run_info(run_dir)
    normal_idx = classes.index('Normal')
    base = dict(top_k_fraction=cfg.mil_top_k_fraction, threshold=cfg.mil_anomaly_threshold)

    grid = [('mil (referencia)', 'mil', base)]
    grid += [(f'mil k={k:.0%}', 'mil', {**base, 'top_k_fraction': k}) for k in TOP_K if k != base['top_k_fraction']]
    grid += [(f'mil umbral={t:.2f}', 'mil', {**base, 'threshold': t}) for t in THRESHOLDS if t != base['threshold']]
    grid += [('voto de todas las ventanas', 'vote_all', {}), ('probabilidad media', 'mean_prob', {}),
             ('max-pooling', 'max_pool', {'threshold': base['threshold']})]

    rows = []
    for split in args.splits:
        if not (run_dir / f'preds_{split}.npz').exists():
            print(f'(sin preds_{split}.npz en {run_dir}; se omite {split})')
            continue
        probs, preds, labels, recs = load_preds(run_dir, split)
        for label, kind, params in grid:
            m = video_metrics(kind, recs, probs, preds, labels, classes, normal_idx, **params)
            rows.append({'split': split, 'scoring': label, **{k: m[k] for k, _ in KEYS}})
        if args.shuffle_frames and split in args.splits:
            sp, spred, slab, srecs = shuffled_predictions(run_dir, cfg, classes, sources, split)
            m = video_metrics('mil', srecs, sp, spred, slab, classes, normal_idx, **base)
            rows.append({'split': split, 'scoring': 'mil + fotogramas barajados', **{k: m[k] for k, _ in KEYS}})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f'scoring_{run_dir.parent.name}_{run_dir.name}'
    with (RESULTS_DIR / f'{stem}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'scoring'] + [k for k, _ in KEYS], lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    lines = [f'# Puntuación por video — `{run_dir.name}`', '',
             '| Split | Puntuación | ' + ' | '.join(lbl for _, lbl in KEYS) + ' |', '|' + '---|' * (len(KEYS) + 2)]
    for r in rows:
        lines.append(f"| {r['split']} | {r['scoring']} | " + ' | '.join(f'{r[k]:.3f}' for k, _ in KEYS) + ' |')
    (RESULTS_DIR / f'{stem}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f"\nGuardado en {RESULTS_DIR / (stem + '.csv')}")


if __name__ == "__main__":
    main()
