"""
0_check_data.py — PASO 0: comprobaciones del dataset antes de lanzar la suite.

1. Cuántos videos y ventanas hay por clase y por split (el conteo del test se publica antes de empezar,
   porque con pocos videos por clase el AUC por clase es solo descriptivo).
2. Resolución de los clips: A13 entrena a 224×224, y subir de resolución un clip más chico no tiene sentido,
   así que solo se justifica si los clips miden al menos 224 px de lado.

Uso
---
    python 0_check_data.py
    python 0_check_data.py --data-root ../data/peru-crime5 --seq-len 16 --stride 16
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 5-ablation/
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))   # raíz de workspace/ (run_report.py)
from run_report import RunReport, current

import cv2
import numpy as np

from ablation import STAGE_DIR
from lib.cli import resolve_data_root
from lib.dataset import _list_videos, discover_classes, index_video, list_sources


def main():
    parser = argparse.ArgumentParser(description="Comprobaciones del dataset de la ablación")
    parser.add_argument("--data-root", default="../data/peru-crime5")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=70)
    parser.add_argument("--min-side", type=int, default=224, help="Lado mínimo requerido para A13")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    sources = list_sources(root)
    classes = discover_classes(root, sources, split="train")
    print(f"Dataset: {root}\nFuentes: {sources}\nClases : {classes}\n")

    sizes = []
    counts = {}
    for split in ("train", "val", "test"):
        print(f"[{split}]  {'clase':<16} {'videos':>7} {'ventanas':>9}")
        tv = tw = 0
        for ci, cls in enumerate(classes):
            vids = [v for s in sources for v in _list_videos(root / s / split / cls)]
            wins = sum(len(index_video(v, ci, args.seq_len, args.stride, frame_stride=args.frame_stride,
                                       max_windows=args.max_windows)) for v in vids)
            tv, tw = tv + len(vids), tw + wins
            counts[f"{split}.{cls}.videos"], counts[f"{split}.{cls}.windows"] = len(vids), wins
            print(f"       {cls:<16} {len(vids):>7} {wins:>9}")
            if split == "test":
                for v in vids:
                    cap = cv2.VideoCapture(str(v))
                    sizes.append((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cls, v.name))
                    cap.release()
        print(f"       {'total':<16} {tv:>7} {tw:>9}\n")

    # resolución: se mira el test (los mismos videos para todas las variantes); train y val vienen del mismo pipeline
    mins = np.array([min(w, h) for w, h, _, _ in sizes])
    print(f"Resolución de los videos de test (lado menor): mín {mins.min()} · mediana {int(np.median(mins))} · máx {mins.max()}")
    small = [(c, n, w, h) for w, h, c, n in sizes if min(w, h) < args.min_side]
    print(f"Videos de test con lado menor < {args.min_side}: {len(small)} de {len(sizes)}")
    for c, n, w, h in small[:10]:
        print(f"   {c}/{n}  {w}×{h}")
    print("\nA13 (224×224):", "viable" if not small else f"revisar: {len(small)} videos de test miden menos de {args.min_side} px")
    current().add_input("data_root", root)
    current().add_results(counts, section="conteos")
    current().add_results({"test_min_side": int(mins.min()), "test_median_side": int(np.median(mins)),
                           f"test_videos_below_{args.min_side}px": len(small), "test_videos": len(sizes)})


if __name__ == "__main__":
    with RunReport(Path(__file__).resolve().parent, "0_check_data"):
        main()
