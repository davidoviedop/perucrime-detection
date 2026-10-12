"""
2_build_rknn.py — PASO 2 (entorno del TOOLKIT, ~/venvs/rknn): convierte backbone.onnx a .rknn y,
con el SIMULADOR de RKNN-Toolkit2, calcula el embedding de cada fotograma de test.

Solo usa numpy y rknn (no importa torch/timm/sklearn). Sin placa: el simulador corre en CPU y
no existe forma documentada de usar GPU con él.

Qué produce, en artifacts/{run_id}/ (una vez por variante):
  backbone_<variante>.rknn          el modelo convertido (su tamaño se reporta)
  build_info_<variante>.json        plataforma, cuantización, algoritmo, versión del toolkit, tiempos
  logs/build_<variante>.log         log detallado de la conversión (aquí aparecen los avisos de
                                    operadores no soportados o reescritos)
  emb_<variante>.npy                embeddings (N,D) float32 de los fotogramas de test, del simulador
  accuracy_<variante>/              (con --accuracy-analysis) error por capa fp32 vs cuantizado

Variantes: fp16, w8a8, w8a8_hybrid, w4a16 (ver rknn_export/rknn_build.py). Por defecto fp16 y w8a8.

Lo que el simulador SÍ da: si el modelo convierte, su tamaño, y los valores numéricos (con la
cuantización simulada). Lo que NO da: velocidad en la NPU ni qué capas corren en NPU frente a CPU
(eso exige una placa).

La simulación de ~3 400 fotogramas puede tardar mucho. Antes de lanzarla completa, mide:
    python 2_build_rknn.py --export-dir <dir> --variants w8a8 --probe
que construye, simula 8 fotogramas, imprime los segundos por fotograma y la estimación total, y sale.
Es reanudable: los trozos ya calculados no se repiten. Con --workers N se reparten entre N procesos
(cada uno construye su propio modelo; la construcción cuantizada también tarda).

Uso
---
    python 2_build_rknn.py --export-dir artifacts/<id> --probe
    python 2_build_rknn.py --export-dir artifacts/<id> --variants fp16 w8a8 --workers 4
    python 2_build_rknn.py --export-dir artifacts/<id> --variants w8a8 --skip-simulate --accuracy-analysis 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 4-export/

import numpy as np

from rknn_export.meta import load_meta
from rknn_export.rknn_build import (VARIANTS, assemble, check, make_rknn, plan_chunks, probe,
                                    run_chunks_parallel, run_chunks_sequential, toolkit_version)

DEFAULT_TARGET = "rv1126b"


def main():
    parser = argparse.ArgumentParser(description="ONNX -> RKNN y embeddings en el simulador")
    parser.add_argument("--export-dir", type=Path, required=True, help="artifacts/{run_id} creada por 1_export_onnx.py")
    parser.add_argument("--variants", nargs="+", default=["fp16", "w8a8"], choices=sorted(VARIANTS),
                        help="Variantes a construir (por defecto: fp16 w8a8)")
    parser.add_argument("--target-platform", default=DEFAULT_TARGET,
                        help=f"Chip destino de rknn.config (por defecto: {DEFAULT_TARGET}); también rk3576, rk3562, rk3588…")
    parser.add_argument("--algorithm", default="normal", choices=["normal", "mmse", "kl_divergence"],
                        help="Algoritmo de cuantización (por defecto: normal, el más rápido; mmse suele ser más preciso y mucho más lento)")
    parser.add_argument("--calib-limit", type=int, default=None,
                        help="Usa solo las primeras N imágenes de calibración (mmse recomienda 20-50)")
    parser.add_argument("--hybrid-cos-thresh", type=float, default=0.98,
                        help="Umbral de coseno de la cuantización híbrida automática (por defecto: 0.98)")
    parser.add_argument("--flash-attention", action="store_true", help="Activa enable_flash_attention en la conversión")
    parser.add_argument("--opt-level", type=int, default=3, choices=[0, 1, 2, 3], help="optimization_level (por defecto: 3)")
    parser.add_argument("--quiet", action="store_true", help="Sin log detallado de la conversión")
    parser.add_argument("--skip-simulate", action="store_true", help="Solo construye y exporta el .rknn")
    parser.add_argument("--probe", action="store_true",
                        help="Mide los segundos por fotograma del simulador y estima el total; no guarda embeddings")
    parser.add_argument("--workers", type=int, default=1, help="Procesos del simulador en paralelo (por defecto: 1)")
    parser.add_argument("--chunk-size", type=int, default=256, help="Fotogramas por trozo reanudable (por defecto: 256)")
    parser.add_argument("--accuracy-analysis", type=int, default=0, metavar="N",
                        help="Con N>0, corre accuracy_analysis sobre N imágenes de calibración (error por capa); solo variantes cuantizadas")
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    meta = load_meta(export_dir)
    onnx_path = export_dir / meta["backbone_onnx"]
    frames_path = export_dir / "test_frames.npy"
    need_frames = not args.skip_simulate
    if need_frames and not frames_path.exists():
        raise SystemExit(f"Falta {frames_path}: ejecuta 1_export_onnx.py sin --skip-frames (o usa --skip-simulate)")

    dataset_txt = None
    if any(VARIANTS[v]["quant"] for v in args.variants):
        dataset_txt = Path(meta["dataset_txt"])
        if args.calib_limit:
            lines = dataset_txt.read_text().splitlines()[:args.calib_limit]
            dataset_txt = export_dir / f"dataset_{args.calib_limit}.txt"
            dataset_txt.write_text("\n".join(lines) + "\n")
        n_calib = len(dataset_txt.read_text().splitlines())
        print(f"Calibración: {n_calib} imágenes ({dataset_txt.name})")

    (export_dir / "logs").mkdir(exist_ok=True)
    frames = np.load(frames_path, mmap_mode="r") if frames_path.exists() else None

    for variant in args.variants:
        spec = VARIANTS[variant]
        print(f"\n=== {variant}: {spec['desc']}  (plataforma {args.target_platform}) ===")
        build_kwargs = dict(
            onnx_path=onnx_path, meta=meta, variant=variant, target_platform=args.target_platform,
            dataset_txt=dataset_txt if spec["quant"] else None, algorithm=args.algorithm,
            opt_level=args.opt_level, flash_attention=args.flash_attention,
            hybrid_cos_thresh=args.hybrid_cos_thresh, verbose=not args.quiet,
            log_file=export_dir / "logs" / f"build_{variant}.log")

        t0 = time.perf_counter()
        rknn = make_rknn(**build_kwargs)
        build_s = time.perf_counter() - t0
        print(f"construido en {build_s:.0f} s")

        rknn_path = export_dir / f"backbone_{variant}.rknn"
        check(rknn.export_rknn(str(rknn_path)), "export_rknn")
        size_mb = rknn_path.stat().st_size / 1e6
        print(f"{rknn_path.name}: {size_mb:.2f} MB")

        info = {
            "variant": variant, "target_platform": args.target_platform, "rknn_file": rknn_path.name,
            "rknn_size_mb": size_mb, "quantized_dtype": spec.get("quantized_dtype", "none"),
            "quantized_algorithm": args.algorithm if spec["quant"] else "none",
            "auto_hybrid": bool(spec.get("auto_hybrid")), "n_calibration": n_calib if spec["quant"] else 0,
            "optimization_level": args.opt_level, "enable_flash_attention": args.flash_attention,
            "build_seconds": round(build_s, 1), "toolkit_version": toolkit_version(),
        }
        (export_dir / f"build_info_{variant}.json").write_text(json.dumps(info, indent=2))

        if args.accuracy_analysis > 0 and spec["quant"]:
            imgs = dataset_txt.read_text().splitlines()[:args.accuracy_analysis]
            out = export_dir / f"accuracy_{variant}"
            print(f"accuracy_analysis sobre {len(imgs)} imágenes -> {out}")
            check(rknn.accuracy_analysis(inputs=imgs, output_dir=str(out)), "accuracy_analysis")

        if args.skip_simulate and not args.probe:
            rknn.release()
            continue

        # ── Simulador ───────────────────────────────────────────────────────
        n_frames = len(frames)
        chunks = plan_chunks(n_frames, args.chunk_size)
        chunk_dir = export_dir / "emb_cache" / variant

        if args.probe:
            check(rknn.init_runtime(), "init_runtime (simulador)")
            sec = probe(rknn, frames, n=8)
            print(f"simulador: {sec:.2f} s por fotograma  →  {n_frames} fotogramas ≈ {sec * n_frames / 3600:.1f} h "
                  f"con 1 proceso (≈ {sec * n_frames / 3600 / max(args.workers, 1):.1f} h con {args.workers} en paralelo, sin contar la construcción de cada uno)")
            rknn.release()
            continue

        if args.workers > 1:
            rknn.release()               # los procesos hijos construyen cada uno el suyo
            settings = {"build": build_kwargs, "frames_path": str(frames_path), "chunk_dir": str(chunk_dir)}
            run_chunks_parallel(settings, chunks, chunk_dir, args.workers)
        else:
            check(rknn.init_runtime(), "init_runtime (simulador)")
            run_chunks_sequential(rknn, frames, chunks, chunk_dir)
            rknn.release()

        emb = assemble(chunk_dir, chunks, export_dir / f"emb_{variant}.npy")
        print(f"emb_{variant}.npy: {emb.shape}")

    print(f"\nSiguiente paso (entorno del proyecto): python 3_score.py --export-dir {export_dir}")


if __name__ == "__main__":
    main()
