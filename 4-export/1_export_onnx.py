"""
1_export_onnx.py — PASO 1 (entorno del PROYECTO): exporta un modelo entrenado a ONNX y prepara
todo lo que necesitan los pasos siguientes.

Entrada: una corrida entrenada (best.pt + run.yaml), p. ej. el estudiante destilado de la etapa 3
(../3-optimization/outputs/<id>) o el profesor de la etapa 2.

Salida, en artifacts/{run_id}/:
  backbone.onnx        backbone por fotograma, entrada (1,3,H,W) normalizada -> (1,D). Va a la NPU.
  temporal_head.onnx   transformer temporal + cabeza, (1,T,D) -> (1,n_clases). Va a la CPU.
  calibration/         imágenes PNG (de ENTRENAMIENTO) para calibrar la cuantización + dataset.txt
  test_frames.npy      fotogramas únicos de test, uint8 RGB (N,H,W,3), ya redimensionados
  test_windows.npz     para cada ventana de test, las filas de test_frames.npy que la forman + su etiqueta
  test_records.json    (video, inicio, etiqueta) de cada ventana, para agrupar por video (MIL)
  export_meta.json     clases, tamaños, normalización… (lo leen los pasos 2 y 3)

Los dos grafos tienen forma estática y lote 1. Tras exportar se compara cada uno con PyTorch
sobre una entrada aleatoria; si la diferencia máxima supera --tol, se detiene.

Uso
---
    python 1_export_onnx.py --run-dir ../3-optimization/outputs/<id_estudiante>
    python 1_export_onnx.py --run-dir <run> --skip-frames          # solo los ONNX (sin fotogramas)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 4-export/ (agrega también 2-modelling/)

import json

import numpy as np
import torch

from rknn_export import ARTIFACTS_DIR
from rknn_export.frames import build_frame_cache, calibration_images, save_calibration, window_records
from rknn_export.meta import (BACKBONE_IN, BACKBONE_OUT, HEAD_IN, HEAD_OUT, MEAN_255, STD_255,
                              save_meta)
from rknn_export.onnx_export import export_backbone, export_temporal_head

from lib.cli import add_override_args, parse_overrides, resolve_data_root
from lib.dataset import discover_classes, list_sources
from lib.model import build_model
from lib.runs import load_run_info


def main():
    parser = argparse.ArgumentParser(description="Exporta un modelo entrenado a ONNX (backbone + temporal/cabeza)")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Carpeta de la corrida entrenada (con best.pt y run.yaml)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Carpeta de salida (por defecto: artifacts/{run_id})")
    parser.add_argument("--opset", type=int, default=17,
                        help="Versión de opset de ONNX (por defecto: 17; RKNN-Toolkit2 2.3.2 admite hasta 19)")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Diferencia máxima tolerada entre PyTorch y ONNX (por defecto: 1e-3)")
    parser.add_argument("--calib-per-class", type=int, default=20,
                        help="Imágenes de calibración por clase (por defecto: 20 → 100 en total; con el algoritmo "
                             "'normal' RKNN recomienda 20-100 y no mejora con más)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-frames", action="store_true",
                        help="Exporta solo los ONNX (omite calibración y fotogramas de test)")
    add_override_args(parser)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cfg, classes, sources = load_run_info(run_dir)
    for key, value in parse_overrides(args).items():
        if not hasattr(cfg, key):
            raise SystemExit(f"campo desconocido en --set: {key}")
        setattr(cfg, key, tuple(value) if isinstance(value, list) else value)

    run_id = run_dir.name
    out_dir = (args.out_dir or ARTIFACTS_DIR / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Corrida   : {run_id}  ({cfg.backbone})")
    print(f"Salida    : {out_dir}")

    # ── Modelo entrenado, en CPU y fp32 ─────────────────────────────────────
    model = build_model(cfg, len(classes), pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    feat_dim = model.backbone.out_dim
    img_size = tuple(cfg.img_size)

    # ── ONNX ────────────────────────────────────────────────────────────────
    backbone_onnx = out_dir / "backbone.onnx"
    head_onnx = out_dir / "temporal_head.onnx"
    d_backbone = export_backbone(model, backbone_onnx, img_size, args.opset)
    d_head = export_temporal_head(model, head_onnx, cfg.seq_len, feat_dim, args.opset)
    print(f"backbone.onnx       {backbone_onnx.stat().st_size / 1e6:7.2f} MB   |PyTorch - ONNX| máx = {d_backbone:.2e}")
    print(f"temporal_head.onnx  {head_onnx.stat().st_size / 1e6:7.2f} MB   |PyTorch - ONNX| máx = {d_head:.2e}")
    if max(d_backbone, d_head) > args.tol:
        raise SystemExit(f"La exportación a ONNX difiere de PyTorch más de {args.tol}: no se sigue.")

    meta = {
        "run_id": run_id, "run_dir": str(run_dir), "backbone": cfg.backbone, "classes": classes,
        "normal_idx": classes.index("Normal") if "Normal" in classes else -1,
        "seq_len": cfg.seq_len, "img_size": list(img_size), "feat_dim": int(feat_dim),
        "mean_255": MEAN_255, "std_255": STD_255, "opset": args.opset,
        "backbone_onnx": backbone_onnx.name, "temporal_head_onnx": head_onnx.name,
        "backbone_io": [BACKBONE_IN, BACKBONE_OUT], "head_io": [HEAD_IN, HEAD_OUT],
        "onnx_max_abs_diff": {"backbone": d_backbone, "temporal_head": d_head},
    }

    # ── Calibración y fotogramas de test ────────────────────────────────────
    if not args.skip_frames:
        data_root = resolve_data_root(cfg.data_root)
        sources = list_sources(data_root)
        classes_data = discover_classes(data_root, sources, split="train")
        if classes_data != classes:
            raise SystemExit(f"Las clases del dataset {classes_data} no coinciden con las de la corrida {classes}")

        train_recs = window_records(data_root, sources, classes, "train", cfg)
        calib = calibration_images(train_recs, cfg, args.calib_per_class, args.seed)
        dataset_txt = save_calibration(calib, out_dir / "calibration")
        print(f"calibración         {len(calib)} imágenes -> {dataset_txt}")

        test_recs = window_records(data_root, sources, classes, "test", cfg)
        frames, window_index = build_frame_cache(test_recs, cfg)
        np.save(out_dir / "test_frames.npy", frames)
        np.savez(out_dir / "test_windows.npz", window_index=window_index,
                 labels=np.array([r[2] for r in test_recs], dtype=np.int64))
        (out_dir / "test_records.json").write_text(json.dumps([list(r) for r in test_recs]))
        print(f"test                {len(test_recs)} ventanas, {len(frames)} fotogramas únicos "
              f"({len(test_recs) * cfg.seq_len / len(frames):.2f}× menos que sin reutilizar)")
        meta.update({"dataset_txt": str(dataset_txt), "n_calibration": len(calib),
                     "n_test_frames": int(len(frames)), "n_test_windows": len(test_recs)})

    save_meta(out_dir, meta)
    print(f"\nexport_meta.json listo. Siguiente paso (entorno del toolkit):\n"
          f"  source ~/venvs/rknn/bin/activate && python 2_build_rknn.py --export-dir {out_dir}")


if __name__ == "__main__":
    main()
