"""
quant_check.py — ¿qué backbones sobreviven a la cuantización int8 de RKNN?  (aparte del flujo 1→2→3)

Para cada backbone candidato (nombre de timm), con pesos preentrenados de ImageNet y SIN entrenar:
  1. lo exporta a ONNX (mismo exportador que 1_export_onnx.py) y comprueba PyTorch vs ONNX;
  2. lo convierte a RKNN en cada construcción pedida (fp16, w8a8, w8a8_kl…);
  3. calcula en el simulador el embedding de N fotogramas de test y lo compara con PyTorch fp32.

La métrica es la similitud coseno del embedding (mismo criterio que 3_score.py): ~1.0 = la
cuantización no daña el backbone; en 8ac40pob (mobilevit_xs entrenado) w8a8 dio 0.135.
No hace falta destilar ni entrenar nada para medirla, así que sirve para DECIDIR a qué backbone
vale la pena destilar el estudiante.

Entorno: el del toolkit (rknn/), que ya tiene torch + timm + rknn juntos. Reutiliza de una
exportación del paso 1 las imágenes de calibración y los fotogramas de test (--export-dir), así
que no necesita el dataset. Necesita red para bajar los pesos preentrenados la primera vez.

Límites de la prueba (conviene tenerlos presentes al leer el resultado):
  - Los pesos son los de ImageNet, no los afinados en tu tarea: los rangos de los pesos pueden
    cambiar al afinar/destilar. Es un indicador de qué familias cuantizan bien, no el número final.
  - Solo mide el backbone (el bloque temporal + cabeza van en CPU y no se cuantizan).
  - No mide velocidad en la NPU ni qué capas caen a CPU (eso exige una placa).

Salida: results/quant_check.csv (una fila por (modelo, construcción); se reemplazan al repetir) y
artifacts/quant_check/<modelo>/ con el ONNX y el .rknn de cada construcción.

Uso
---
    python quant_check.py --export-dir artifacts/8ac40pob
    python quant_check.py --export-dir artifacts/8ac40pob --models resnet18 mobilenetv3_large_100 --builds fp16 w8a8
    python quant_check.py --export-dir artifacts/8ac40pob --n-frames 100        # más rápido
"""
import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 4-export/
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))   # raíz de workspace/ (run_report.py)
from run_report import RunReport, current

import numpy as np
import timm
import torch

from rknn_export import ARTIFACTS_DIR, RESULTS_DIR
from rknn_export.meta import MEAN_255, STD_255, load_meta
from rknn_export.onnx_export import export_backbone
from rknn_export.rknn_build import check, infer_embedding, make_rknn, toolkit_version
from rknn_export.scoring import _normalize, embedding_fidelity

# Candidatos por defecto: los dos que ya se pensaba probar (resnet, mobilenet), el backbone de la
# línea base de la ablación (ConvNeXtV2-Femto) y mobilevit_xs como control (conocido: se rompe).
DEFAULT_MODELS = ["resnet18", "mobilenetv2_100", "mobilenetv3_large_100", "convnextv2_femto", "mobilevit_xs"]

# nombre -> (variante base de rknn_build.VARIANTS, algoritmo, imágenes de calibración o None = todas)
BUILDS = {
    "fp16":        ("fp16",        "normal",        None),
    "w8a8":        ("w8a8",        "normal",        None),
    "w8a8_kl":     ("w8a8",        "kl_divergence", None),
    "w8a8_mmse":   ("w8a8",        "mmse",          50),
    "w8a8_hybrid": ("w8a8_hybrid", "normal",        None),
}
DEFAULT_BUILDS = ["fp16", "w8a8", "w8a8_kl"]

COLUMNS = ["model", "build", "params_M", "feat_dim", "onnx_max_diff", "rknn_size_mb", "n_frames",
           "emb_cos_mean", "emb_cos_p05", "emb_relL2_mean", "target_platform", "build_seconds",
           "toolkit_version", "error"]


def pick_frames(export_dir: Path, n: int) -> np.ndarray:
    """n fotogramas de test repartidos por todo el conjunto (no los n primeros: serían de un solo video)."""
    frames = np.load(export_dir / "test_frames.npy", mmap_mode="r")
    idx = np.unique(np.linspace(0, len(frames) - 1, num=min(n, len(frames))).astype(int))
    return np.ascontiguousarray(frames[idx])


def calibration_file(export_dir: Path, limit, work_dir: Path) -> Path:
    dataset_txt = export_dir / "dataset.txt"
    if not limit:
        return dataset_txt
    lines = dataset_txt.read_text().splitlines()[:limit]
    out = work_dir / f"dataset_{limit}.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_rows(path: Path, rows: list):
    """Agrega filas al CSV reemplazando las anteriores del mismo (modelo, construcción)."""
    keys = {(r["model"], r["build"]) for r in rows}
    kept = []
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if (r["model"], r["build"]) not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
        w.writerows(rows)
    os.replace(tmp, path)


def check_model(name, args, export_dir, frames, meta, out_root):
    """Filas (una por construcción) de un backbone. Un fallo se anota en `error` y no detiene al resto."""
    work = out_root / name
    work.mkdir(parents=True, exist_ok=True)
    base = {"model": name, "n_frames": len(frames), "target_platform": args.target_platform,
            "toolkit_version": toolkit_version()}

    # ── PyTorch fp32 (referencia) y ONNX ─────────────────────────────────────
    try:
        backbone = timm.create_model(name, pretrained=not args.random_weights, num_classes=0).eval()
        base["params_M"] = round(sum(p.numel() for p in backbone.parameters()) / 1e6, 3)
        base["feat_dim"] = backbone.num_features
        with torch.no_grad():
            ref = np.concatenate([backbone(torch.from_numpy(_normalize(frames[a:a + 32]))).numpy()
                                  for a in range(0, len(frames), 32)])
        onnx_path = work / "backbone.onnx"
        base["onnx_max_diff"] = export_backbone(SimpleNamespace(backbone=backbone), onnx_path,
                                                tuple(meta["img_size"]), args.opset)
        print(f"  {name}: {base['params_M']} M parámetros, embedding {base['feat_dim']}, "
              f"|PyTorch - ONNX| = {base['onnx_max_diff']:.2e}")
    except Exception as e:                                    # sin red, nombre inexistente, exportación…
        msg = f"{type(e).__name__}: {e}"
        print(f"  {name}: FALLÓ antes de convertir — {msg[:200]}")
        return [{**base, "build": b, "error": f"preparación: {msg[:300]}"} for b in args.builds]

    rows = []
    for build in args.builds:
        variant, algorithm, calib = BUILDS[build]
        row = {**base, "build": build}
        t0 = time.perf_counter()
        try:
            quant = variant != "fp16"
            rknn = make_rknn(
                onnx_path=onnx_path, meta=meta, variant=variant, target_platform=args.target_platform,
                dataset_txt=calibration_file(export_dir, calib, work) if quant else None,
                algorithm=algorithm, verbose=False)
            row["build_seconds"] = round(time.perf_counter() - t0, 1)
            rknn_path = work / f"backbone_{build}.rknn"
            check(rknn.export_rknn(str(rknn_path)), "export_rknn")
            row["rknn_size_mb"] = round(rknn_path.stat().st_size / 1e6, 3)
            check(rknn.init_runtime(), "init_runtime (simulador)")
            emb = np.stack([infer_embedding(rknn, f) for f in frames])
            rknn.release()
            row.update(embedding_fidelity(ref, emb))
            print(f"    {build:12s} cos={row['emb_cos_mean']:.4f}  p05={row['emb_cos_p05']:.3f}  "
                  f"{row['rknn_size_mb']:.2f} MB  (construcción {row['build_seconds']:.0f} s)")
        except BaseException as e:                            # SystemExit de check() incluido
            if isinstance(e, KeyboardInterrupt):
                raise
            msg = f"{type(e).__name__}: {e}"
            row["error"] = msg[:300]
            print(f"    {build:12s} FALLÓ — {msg[:160]}")
            if args.traceback:
                traceback.print_exc()
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fidelidad int8 de RKNN para varios backbones (sin entrenar)")
    parser.add_argument("--export-dir", type=Path, required=True,
                        help="artifacts/{run_id} de 1_export_onnx.py (aporta calibración y fotogramas de test)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, metavar="TIMM_NAME",
                        help=f"Backbones de timm (por defecto: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--builds", nargs="+", default=DEFAULT_BUILDS, choices=sorted(BUILDS),
                        help=f"Construcciones a probar (por defecto: {' '.join(DEFAULT_BUILDS)}); mmse es muy lento")
    parser.add_argument("--n-frames", type=int, default=200,
                        help="Fotogramas de test para comparar embeddings (por defecto: 200)")
    parser.add_argument("--target-platform", default="rv1126b")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--random-weights", action="store_true",
                        help="Sin pesos preentrenados (no necesita red, pero los rangos de los pesos no son realistas)")
    parser.add_argument("--traceback", action="store_true", help="Imprime la traza completa de los fallos")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "quant_check.csv")
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    meta = load_meta(export_dir)
    # Las construcciones solo necesitan tamaño de imagen y normalización, que son las del entrenamiento
    meta = {"img_size": meta["img_size"], "mean_255": MEAN_255, "std_255": STD_255}
    for f in ("dataset.txt", "test_frames.npy"):
        if not (export_dir / f).exists():
            raise SystemExit(f"Falta {export_dir / f}: ejecuta 1_export_onnx.py sin --skip-frames")

    frames = pick_frames(export_dir, args.n_frames)
    out_root = ARTIFACTS_DIR / "quant_check"
    current().add_input("export_dir", export_dir)
    print(f"{len(args.models)} backbones × {len(args.builds)} construcciones, "
          f"{len(frames)} fotogramas, imagen {meta['img_size']}, plataforma {args.target_platform}")

    all_rows = []
    for name in args.models:
        print(f"\n=== {name} ===")
        rows = check_model(name, args, export_dir, frames, meta, out_root)
        write_rows(args.out, rows)                            # se guarda tras cada modelo: un corte no pierde lo hecho
        all_rows += rows

    print(f"\n{'modelo':24s} {'build':12s} {'cos':>7s} {'p05':>7s} {'MB':>7s}")
    for r in all_rows:
        if r.get("error"):
            print(f"{r['model']:24s} {r['build']:12s}  FALLÓ: {r['error'][:60]}")
        else:
            print(f"{r['model']:24s} {r['build']:12s} {r['emb_cos_mean']:7.4f} {r['emb_cos_p05']:7.3f} {r['rknn_size_mb']:7.2f}")
    print(f"\nGuardado en {args.out}")
    current().add_results({f"{r['model']}.{r['build']}.{k}": r[k] for r in all_rows
                           for k in ("emb_cos_mean", "emb_cos_p05", "rknn_size_mb") if k in r}, section="quant_check")
    current().add_output(args.out)


if __name__ == "__main__":
    with RunReport(Path(__file__).resolve().parent, "quant_check"):
        main()
