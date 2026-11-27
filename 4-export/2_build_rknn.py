"""
2_build_rknn.py — PASO 2 (entorno del TOOLKIT, ~/venvs/rknn): convierte backbone.onnx a .rknn y,
con el SIMULADOR de RKNN-Toolkit2, calcula el embedding de cada fotograma de test.

Solo usa numpy y rknn (no importa torch/timm/sklearn). Sin placa: el simulador corre en CPU y
no existe forma documentada de usar GPU con él.

Qué construye lo define builds.yaml: cada entrada de `builds` es una construcción con nombre
(variante base + algoritmo, calibración, umbral híbrido…), así se prueban varias opciones de
cuantización sin pisarse. Por defecto se construyen las de `run:`; --builds elige otras.

Qué produce, en artifacts/{run_id}/ (una vez por construcción <nombre>):
  backbone_<nombre>.rknn            el modelo convertido (su tamaño se reporta)
  build_info_<nombre>.json          plataforma, cuantización, algoritmo, versión del toolkit, tiempos
  logs/build_<nombre>.log           log detallado de la conversión (aquí aparecen los avisos de
                                    operadores no soportados o reescritos)
  emb_<nombre>.npy                  embeddings (N,D) float32 de los fotogramas de test, del simulador
  accuracy_<nombre>/                (con --accuracy-analysis) error por capa fp32 vs cuantizado

Variantes base: fp16, w8a8, w8a8_hybrid, w4a16 (ver rknn_export/rknn_build.py).

Lo que el simulador SÍ da: si el modelo convierte, su tamaño, y los valores numéricos (con la
cuantización simulada). Lo que NO da: velocidad en la NPU ni qué capas corren en NPU frente a CPU
(eso exige una placa).

La simulación de ~3 400 fotogramas puede tardar mucho. Antes de lanzarla completa, mide:
    python 2_build_rknn.py --export-dir <dir> --builds w8a8 --probe
que construye, simula 8 fotogramas, imprime los segundos por fotograma y la estimación total, y sale.
Es reanudable: los trozos ya calculados no se repiten. Con --workers N se reparten entre N procesos
(cada uno construye su propio modelo; la construcción cuantizada también tarda).

Uso
---
    python 2_build_rknn.py --export-dir artifacts/<id> --builds w8a8 --probe
    python 2_build_rknn.py --export-dir artifacts/<id>                         # las de `run:` en builds.yaml
    python 2_build_rknn.py --export-dir artifacts/<id> --builds w8a8_mmse w8a8_kl --workers 4
    python 2_build_rknn.py --export-dir artifacts/<id> --builds w8a8 --skip-simulate --accuracy-analysis 8
    # --algorithm / --calib-limit / ... pisan lo que diga el YAML en TODAS las construcciones elegidas
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 4-export/
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))   # raíz de workspace/ (run_report.py)
from run_report import RunReport, current

import numpy as np
import yaml

from rknn_export.meta import load_meta
from rknn_export.rknn_build import (VARIANTS, assemble, check, make_rknn, plan_chunks, prepare_chunk_dir,
                                    probe, run_chunks_parallel, run_chunks_sequential, toolkit_version)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "builds.yaml"
ALGORITHMS = ["normal", "mmse", "kl_divergence"]
BUILD_KEYS = {"variant", "algorithm", "quantized_method", "calib_limit", "hybrid_cos_thresh", "opt_level",
              "flash_attention"}


def load_builds(config_path: Path, names, overrides: dict):
    """(target_platform, {nombre: parámetros completos}) de builds.yaml.

    `names` None -> las de `run:`. `overrides` (flags de la CLI que no son None) pisan cada build."""
    cfg = yaml.safe_load(config_path.read_text()) or {}
    defined = cfg.get("builds") or {}
    defaults = cfg.get("defaults") or {}
    names = list(names or cfg.get("run") or [])
    if not names:
        raise SystemExit(f"{config_path.name}: no hay construcciones que hacer (ni --builds ni `run:`)")
    unknown = [n for n in names if n not in defined]
    if unknown:
        raise SystemExit(f"Construcciones desconocidas {unknown}; en {config_path.name} hay: {sorted(defined)}")
    builds = {}
    for n in names:
        b = {**defaults, **(defined[n] or {}), **overrides}
        if bad := set(b) - BUILD_KEYS:
            raise SystemExit(f"{config_path.name}: la construcción '{n}' tiene campos desconocidos {sorted(bad)}; "
                             f"válidos: {sorted(BUILD_KEYS)}")
        if b.get("variant") not in VARIANTS:
            raise SystemExit(f"{config_path.name}: '{n}' necesita variant en {sorted(VARIANTS)} (tiene {b.get('variant')!r})")
        if b.get("algorithm", "normal") not in ALGORITHMS:
            raise SystemExit(f"{config_path.name}: '{n}' tiene algorithm {b['algorithm']!r}; válidos: {ALGORITHMS}")
        if b.get("quantized_method", "channel") not in ("channel", "layer"):
            raise SystemExit(f"{config_path.name}: '{n}' tiene quantized_method {b['quantized_method']!r}; válidos: channel, layer")
        builds[n] = {"algorithm": "normal", "quantized_method": "channel", "calib_limit": None,
                     "hybrid_cos_thresh": 0.98, "opt_level": 3, "flash_attention": False, **b}
    return cfg.get("target_platform", "rv1126b"), builds


def main():
    parser = argparse.ArgumentParser(description="ONNX -> RKNN y embeddings en el simulador")
    parser.add_argument("--export-dir", type=Path, required=True, help="artifacts/{run_id} creada por 1_export_onnx.py")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML con las construcciones (por defecto: builds.yaml junto al script)")
    parser.add_argument("--builds", nargs="+", default=None, metavar="NOMBRE",
                        help="Construcciones de builds.yaml a hacer (por defecto: las de `run:`)")
    parser.add_argument("--target-platform", default=None,
                        help="Pisa target_platform del YAML (rv1126b, rk3576, rk3562, rk3588…)")
    parser.add_argument("--algorithm", default=None, choices=ALGORITHMS,
                        help="Pisa el algoritmo de cuantización de todas las construcciones elegidas")
    parser.add_argument("--quantized-method", default=None, choices=["channel", "layer"],
                        help="Pisa quantized_method de todas las construcciones elegidas")
    parser.add_argument("--calib-limit", type=int, default=None,
                        help="Pisa calib_limit: usa solo las primeras N imágenes de calibración")
    parser.add_argument("--hybrid-cos-thresh", type=float, default=None,
                        help="Pisa el umbral de coseno de la cuantización híbrida")
    parser.add_argument("--flash-attention", action="store_const", const=True, default=None,
                        help="Activa enable_flash_attention en todas las construcciones elegidas")
    parser.add_argument("--opt-level", type=int, default=None, choices=[0, 1, 2, 3], help="Pisa optimization_level")
    parser.add_argument("--quiet", action="store_true", help="Sin log detallado de la conversión")
    parser.add_argument("--skip-simulate", action="store_true", help="Solo construye y exporta el .rknn")
    parser.add_argument("--probe", action="store_true",
                        help="Mide los segundos por fotograma del simulador y estima el total; no guarda embeddings")
    parser.add_argument("--workers", type=int, default=1, help="Procesos del simulador en paralelo (por defecto: 1)")
    parser.add_argument("--chunk-size", type=int, default=256, help="Fotogramas por trozo reanudable (por defecto: 256)")
    parser.add_argument("--accuracy-analysis", type=int, default=0, metavar="N",
                        help="Con N>0, corre accuracy_analysis sobre N imágenes de calibración (error por capa); solo variantes cuantizadas")
    args = parser.parse_args()

    overrides = {k: v for k, v in dict(
        algorithm=args.algorithm, quantized_method=args.quantized_method, calib_limit=args.calib_limit,
        hybrid_cos_thresh=args.hybrid_cos_thresh, flash_attention=args.flash_attention,
        opt_level=args.opt_level).items() if v is not None}
    cfg_target, builds = load_builds(args.config.resolve(), args.builds, overrides)
    target = args.target_platform or cfg_target

    export_dir = args.export_dir.resolve()
    meta = load_meta(export_dir)
    current().set_run_id(meta['run_id'])
    current().add_input('export_dir', export_dir)
    current().add_input('config', args.config.resolve())
    onnx_path = export_dir / meta["backbone_onnx"]
    frames_path = export_dir / "test_frames.npy"
    need_frames = not args.skip_simulate
    if need_frames and not frames_path.exists():
        raise SystemExit(f"Falta {frames_path}: ejecuta 1_export_onnx.py sin --skip-frames (o usa --skip-simulate)")
    print(f"Construcciones ({args.config.name}): {', '.join(builds)}")

    (export_dir / "logs").mkdir(exist_ok=True)
    frames = np.load(frames_path, mmap_mode="r") if frames_path.exists() else None
    onnx_stat = onnx_path.stat()

    for name, bp in builds.items():
        variant = bp["variant"]
        spec = VARIANTS[variant]
        quant = spec["quant"]
        extra = ""
        if quant:
            extra = f", {bp['algorithm']}, {bp['quantized_method']}" + (f", calib {bp['calib_limit']}" if bp["calib_limit"] else "")
            if spec.get("auto_hybrid"):
                extra += f", cos≥{bp['hybrid_cos_thresh']}"
        print(f"\n=== {name}: {spec['desc']}  [{variant}{extra}]  (plataforma {target}) ===")

        dataset_txt, n_calib = None, 0
        if quant:
            dataset_txt = Path(meta["dataset_txt"])
            if bp["calib_limit"]:
                lines = dataset_txt.read_text().splitlines()[:bp["calib_limit"]]
                dataset_txt = export_dir / f"dataset_{bp['calib_limit']}.txt"
                dataset_txt.write_text("\n".join(lines) + "\n")
            n_calib = len(dataset_txt.read_text().splitlines())
            print(f"calibración: {n_calib} imágenes ({dataset_txt.name})")

        build_kwargs = dict(
            onnx_path=onnx_path, meta=meta, variant=variant, target_platform=target,
            dataset_txt=dataset_txt, algorithm=bp["algorithm"], quantized_method=bp["quantized_method"],
            opt_level=bp["opt_level"], flash_attention=bp["flash_attention"],
            hybrid_cos_thresh=bp["hybrid_cos_thresh"], verbose=not args.quiet,
            log_file=export_dir / "logs" / f"build_{name}.log")

        t0 = time.perf_counter()
        rknn = make_rknn(**build_kwargs)
        build_s = time.perf_counter() - t0
        print(f"construido en {build_s:.0f} s")

        rknn_path = export_dir / f"backbone_{name}.rknn"
        check(rknn.export_rknn(str(rknn_path)), "export_rknn")
        size_mb = rknn_path.stat().st_size / 1e6
        print(f"{rknn_path.name}: {size_mb:.2f} MB")

        info = {
            "build": name, "variant": variant, "target_platform": target, "rknn_file": rknn_path.name,
            "rknn_size_mb": size_mb, "quantized_dtype": spec.get("quantized_dtype", "none"),
            "quantized_algorithm": bp["algorithm"] if quant else "none",
            "quantized_method": bp["quantized_method"] if quant else "none",
            "auto_hybrid": bool(spec.get("auto_hybrid")),
            "hybrid_cos_thresh": bp["hybrid_cos_thresh"] if spec.get("auto_hybrid") else None,
            "n_calibration": n_calib,
            "optimization_level": bp["opt_level"], "enable_flash_attention": bp["flash_attention"],
            "build_seconds": round(build_s, 1), "toolkit_version": toolkit_version(),
        }
        (export_dir / f"build_info_{name}.json").write_text(json.dumps(info, indent=2))
        current().add_results(info, section=f"build_{name}")
        current().add_output(export_dir / f"build_info_{name}.json")
        current().flush()

        if args.accuracy_analysis > 0 and spec["quant"]:
            imgs = dataset_txt.read_text().splitlines()[:args.accuracy_analysis]
            out = export_dir / f"accuracy_{name}"
            print(f"accuracy_analysis sobre {len(imgs)} imágenes -> {out}")
            check(rknn.accuracy_analysis(inputs=imgs, output_dir=str(out)), "accuracy_analysis")

        if args.skip_simulate and not args.probe:
            rknn.release()
            continue

        # ── Simulador ───────────────────────────────────────────────────────
        n_frames = len(frames)
        chunks = plan_chunks(n_frames, args.chunk_size)
        chunk_dir = export_dir / "emb_cache" / name

        if args.probe:
            check(rknn.init_runtime(), "init_runtime (simulador)")
            sec = probe(rknn, frames, n=8)
            print(f"simulador: {sec:.2f} s por fotograma  →  {n_frames} fotogramas ≈ {sec * n_frames / 3600:.1f} h "
                  f"con 1 proceso (≈ {sec * n_frames / 3600 / max(args.workers, 1):.1f} h con {args.workers} en paralelo, sin contar la construcción de cada uno)")
            rknn.release()
            continue

        # Firma de la construcción: si cambió algo, los trozos viejos no se reutilizan
        prepare_chunk_dir(chunk_dir, {**{k: str(v) if isinstance(v, Path) else v for k, v in bp.items()},
                                      "target_platform": target, "n_calibration": n_calib,
                                      "onnx_size": onnx_stat.st_size, "onnx_mtime_ns": onnx_stat.st_mtime_ns})

        if args.workers > 1:
            rknn.release()               # los procesos hijos construyen cada uno el suyo
            settings = {"build": build_kwargs, "frames_path": str(frames_path), "chunk_dir": str(chunk_dir)}
            run_chunks_parallel(settings, chunks, chunk_dir, args.workers)
        else:
            check(rknn.init_runtime(), "init_runtime (simulador)")
            run_chunks_sequential(rknn, frames, chunks, chunk_dir)
            rknn.release()

        emb = assemble(chunk_dir, chunks, export_dir / f"emb_{name}.npy")
        print(f"emb_{name}.npy: {emb.shape}")

    print(f"\nSiguiente paso (entorno del proyecto): python 3_score.py --export-dir {export_dir}")


if __name__ == "__main__":
    with RunReport(Path(__file__).resolve().parent, "2_build_rknn"):
        main()
