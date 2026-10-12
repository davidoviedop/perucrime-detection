"""
3_score.py — PASO 3 (entorno del PROYECTO): puntúa cada variante exportada con las métricas de la tesis.

Para cada variante se toman sus embeddings por fotograma (emb_<variante>.npy, calculados en el
simulador por 2_build_rknn.py), se arman las ventanas de test, se pasan por temporal_head.onnx (que
en el despliegue corre en CPU) y se calculan AUC macro/binario por ventana y por video (MIL).

Referencias que siempre se calculan (sin RKNN):
  torch_fp32   backbone de PyTorch tal como se entrenó
  onnx_fp32    backbone.onnx con onnxruntime (comprueba que la exportación no pierde nada)
Cada variante RKNN se compara contra torch_fp32 (similitud coseno y error L2 relativo por embedding).

Salida: results/rknn_export.csv (una fila por (corrida, variante); se reemplazan al repetir).

Uso
---
    python 3_score.py --export-dir artifacts/<id>
    python 3_score.py --export-dir artifacts/<id> --variants fp16 w8a8 --no-onnx-ref
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 4-export/ (agrega también 2-modelling/)

import numpy as np
import torch

from rknn_export import RESULTS_DIR
from rknn_export.meta import load_meta
from rknn_export.scoring import (embedding_fidelity, embeddings_onnx, embeddings_torch,
                                 logits_from_embeddings, softmax, write_rows)

from lib.evaluation import score_predictions
from lib.model import build_model
from lib.runs import load_run_info


def score_variant(emb, window_index, head_onnx, labels, recs, classes, normal_idx, cfg):
    logits = logits_from_embeddings(emb, window_index, head_onnx)
    probs = softmax(logits)
    return score_predictions(probs, probs.argmax(axis=1), labels, recs, classes, normal_idx, cfg)["flat"]


def main():
    parser = argparse.ArgumentParser(description="Métricas de las variantes exportadas (backbone + cabeza en CPU)")
    parser.add_argument("--export-dir", type=Path, required=True, help="artifacts/{run_id}")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variantes RKNN a puntuar (por defecto: todas las que tengan emb_<variante>.npy)")
    parser.add_argument("--no-onnx-ref", action="store_true", help="Omite la referencia onnx_fp32 (lenta: 1 fotograma por llamada)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "rknn_export.csv")
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    meta = load_meta(export_dir)
    run_dir = Path(meta["run_dir"])
    cfg, classes, _ = load_run_info(run_dir)
    normal_idx = meta["normal_idx"]
    head_onnx = export_dir / meta["temporal_head_onnx"]

    frames = np.load(export_dir / "test_frames.npy", mmap_mode="r")
    win = np.load(export_dir / "test_windows.npz")
    window_index, labels = win["window_index"], win["labels"]
    recs = [tuple(r) for r in json.loads((export_dir / "test_records.json").read_text())]

    variants = args.variants or sorted(p.stem[len("emb_"):] for p in export_dir.glob("emb_*.npy"))
    print(f"Variantes RKNN a puntuar: {variants or 'ninguna'}")

    # ── Referencia PyTorch ──────────────────────────────────────────────────
    model = build_model(cfg, len(classes), pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = embeddings_torch(model.backbone, frames, device)
    np.save(export_dir / "emb_torch_fp32.npy", ref)

    entries = [("torch_fp32", ref, None, None)]
    if not args.no_onnx_ref:
        emb_onnx = embeddings_onnx(export_dir / meta["backbone_onnx"], frames)
        np.save(export_dir / "emb_onnx_fp32.npy", emb_onnx)
        entries.append(("onnx_fp32", emb_onnx, None, None))

    for v in variants:
        info_path = export_dir / f"build_info_{v}.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        entries.append((v, np.load(export_dir / f"emb_{v}.npy"), info, export_dir / f"backbone_{v}.rknn"))

    # ── Puntuación ──────────────────────────────────────────────────────────
    rows = []
    for name, emb, info, rknn_file in entries:
        if len(emb) != len(frames):
            raise SystemExit(f"{name}: {len(emb)} embeddings, se esperaban {len(frames)}")
        flat = score_variant(emb, window_index, head_onnx, labels, recs, classes, normal_idx, cfg)
        fid = embedding_fidelity(ref, emb)
        size_mb = rknn_file.stat().st_size / 1e6 if rknn_file and rknn_file.exists() else \
            (export_dir / meta["backbone_onnx"]).stat().st_size / 1e6
        row = {
            "run_id": meta["run_id"], "backbone": meta["backbone"],
            "target_platform": (info or {}).get("target_platform", "n/a"), "variant": name,
            "backbone_size_mb": round(size_mb, 3), "n_frames": len(emb), **fid,
            **{k: flat[f"test_{k}"] for k in ("window_auc_macro", "window_auc_binary", "video_auc_macro",
                                              "video_auc_binary", "window_accuracy", "video_accuracy")},
            "quantized_dtype": (info or {}).get("quantized_dtype", "none"),
            "quantized_algorithm": (info or {}).get("quantized_algorithm", "none"),
            "toolkit_version": (info or {}).get("toolkit_version", "n/a"),
        }
        rows.append(row)
        print(f"{name:12s} cos={fid['emb_cos_mean']:.4f}  win AUC macro={row['test_window_auc_macro']:.4f} "
              f"bin={row['test_window_auc_binary']:.4f} | video macro={row['test_video_auc_macro']:.4f} "
              f"bin={row['test_video_auc_binary']:.4f}  ({size_mb:.2f} MB)")

    write_rows(args.out, rows, meta["run_id"])
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
