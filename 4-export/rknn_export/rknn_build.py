"""
rknn_export/rknn_build.py — convierte backbone.onnx a .rknn y calcula embeddings en el SIMULADOR.

Este módulo corre en el entorno del toolkit (~/venvs/rknn): usa solo numpy y rknn, nada de
torch/timm/sklearn. Sin placa, el simulador es la única forma de ejecutar el modelo:
`rknn.init_runtime()` sin `target` lo activa.

Dos particularidades del toolkit (v2.3.2, según su referencia de la API):
  - Con `target=None` hay que llamar antes a build(); un modelo cargado con load_rknn() solo
    sirve para hardware, no para el simulador. Por eso cada proceso que quiera simular tiene
    que construir su propio modelo (config → load_onnx → build) en la misma sesión.
  - La normalización (mean/std) la aplica la herramienta antes de la NPU, así que el simulador
    recibe fotogramas crudos uint8 en RGB con forma (1, H, W, 3).

Variantes (todas con la misma plataforma destino):
  fp16         sin cuantizar
  w8a8         pesos y activaciones en int8
  w8a8_hybrid  int8 con las capas que más pierden convertidas a fp16 de forma automática
  w4a16        pesos en 4 bit y activaciones en fp16 (solo RK3576 / RV1126B)
"""
import time
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .meta import BACKBONE_IN

VARIANTS = {
    "fp16":        dict(quant=False, desc="sin cuantizar (float16)"),
    "w8a8":        dict(quant=True, quantized_dtype="w8a8", desc="int8 en pesos y activaciones"),
    "w8a8_hybrid": dict(quant=True, quantized_dtype="w8a8", auto_hybrid=True,
                        desc="int8 con las capas sensibles en fp16 (híbrida automática)"),
    "w4a16":       dict(quant=True, quantized_dtype="w4a16", desc="pesos 4 bit, activaciones fp16"),
}


def toolkit_version() -> str:
    try:
        return metadata.version("rknn-toolkit2")
    except metadata.PackageNotFoundError:
        return "desconocida"


def check(ret, what):
    if ret != 0:
        raise SystemExit(f"RKNN: {what} falló (código {ret}). Revisa el log de la construcción en artifacts/.../logs/")


def make_rknn(onnx_path, meta, variant, target_platform, dataset_txt=None, algorithm="normal",
              opt_level=3, flash_attention=False, hybrid_cos_thresh=0.98,
              verbose=True, log_file=None):
    """config → load_onnx → build. Devuelve el objeto RKNN ya construido (listo para exportar o simular)."""
    from rknn.api import RKNN              # import perezoso: solo existe en el entorno del toolkit

    spec = VARIANTS[variant]
    h, w = meta["img_size"]
    rknn = RKNN(verbose=verbose, verbose_file=str(log_file) if (verbose and log_file) else None)

    cfg = dict(
        mean_values=[meta["mean_255"]], std_values=[meta["std_255"]],
        target_platform=target_platform, optimization_level=opt_level,
        quantized_algorithm=algorithm, enable_flash_attention=flash_attention,
        auto_hybrid_cos_thresh=hybrid_cos_thresh,
    )
    if spec.get("quantized_dtype"):
        cfg["quantized_dtype"] = spec["quantized_dtype"]
    rknn.config(**cfg)

    check(rknn.load_onnx(model=str(onnx_path), inputs=[BACKBONE_IN], input_size_list=[[1, 3, h, w]]),
          "load_onnx")

    build_kwargs = dict(do_quantization=spec["quant"])
    if spec["quant"]:
        build_kwargs["dataset"] = str(dataset_txt)
    if spec.get("auto_hybrid"):
        build_kwargs["auto_hybrid"] = True
    check(rknn.build(**build_kwargs), "build")
    return rknn


def infer_embedding(rknn, frame_hwc_u8: np.ndarray) -> np.ndarray:
    """Embedding (D,) de UN fotograma uint8 RGB (H, W, 3) en el simulador."""
    batch = np.ascontiguousarray(frame_hwc_u8[None, ...])          # (1, H, W, 3), layout nhwc por defecto
    out = rknn.inference(inputs=[batch])
    return np.asarray(out[0], dtype=np.float32).reshape(-1)


def probe(rknn, frames, n=8):
    """Segundos por fotograma del simulador sobre los primeros `n` fotogramas (primero uno de calentamiento)."""
    infer_embedding(rknn, frames[0])
    t0 = time.perf_counter()
    for i in range(n):
        infer_embedding(rknn, frames[i])
    return (time.perf_counter() - t0) / n


# ── Cálculo por trozos, con reanudación ───────────────────────────────────────

def plan_chunks(n_frames: int, size: int):
    """[(índice, inicio, fin), …] que cubren todos los fotogramas."""
    return [(i, a, min(a + size, n_frames)) for i, a in enumerate(range(0, n_frames, size))]


def chunk_path(chunk_dir, i) -> Path:
    return Path(chunk_dir) / f"chunk_{i:04d}.npy"


def run_chunks_sequential(rknn, frames, chunks, chunk_dir, log_every=1):
    """Calcula en este proceso los trozos que aún no están en disco."""
    Path(chunk_dir).mkdir(parents=True, exist_ok=True)
    pending = [c for c in chunks if not chunk_path(chunk_dir, c[0]).exists()]
    t0 = time.perf_counter()
    for k, (i, a, b) in enumerate(pending, 1):
        emb = np.stack([infer_embedding(rknn, frames[j]) for j in range(a, b)])
        np.save(chunk_path(chunk_dir, i), emb)
        if k % log_every == 0:
            el = time.perf_counter() - t0
            print(f"  trozo {k}/{len(pending)}  ({b}/{len(frames)} fotogramas)  "
                  f"{el:.0f}s transcurridos, ~{el / k * (len(pending) - k):.0f}s restantes", flush=True)


_W: dict = {}


def _worker_init(settings):
    """Cada proceso construye su propio modelo (el simulador exige haber llamado a build() en la sesión)."""
    _W["rknn"] = make_rknn(**settings["build"])
    check(_W["rknn"].init_runtime(), "init_runtime (simulador)")
    _W["frames"] = np.load(settings["frames_path"], mmap_mode="r")
    _W["chunk_dir"] = settings["chunk_dir"]


def _worker_run(task):
    i, a, b = task
    emb = np.stack([infer_embedding(_W["rknn"], _W["frames"][j]) for j in range(a, b)])
    np.save(chunk_path(_W["chunk_dir"], i), emb)
    return i


def run_chunks_parallel(settings, chunks, chunk_dir, workers):
    """Reparte los trozos pendientes entre `workers` procesos (cada uno construye su modelo)."""
    Path(chunk_dir).mkdir(parents=True, exist_ok=True)
    pending = [c for c in chunks if not chunk_path(chunk_dir, c[0]).exists()]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=_worker_init, initargs=(settings,)) as pool:
        for k, _ in enumerate(pool.map(_worker_run, pending), 1):
            el = time.perf_counter() - t0
            print(f"  trozo {k}/{len(pending)}  {el:.0f}s transcurridos, "
                  f"~{el / k * (len(pending) - k):.0f}s restantes", flush=True)


def assemble(chunk_dir, chunks, out_path) -> np.ndarray:
    """Une los trozos en orden y guarda emb_<variante>.npy, de forma (n_fotogramas, D) float32."""
    emb = np.concatenate([np.load(chunk_path(chunk_dir, i)) for i, _, _ in chunks], axis=0)
    np.save(out_path, emb.astype(np.float32))
    return emb
