"""
optimization/benchmark.py — costo de un modelo: tamaño, GFLOPs, y velocidad (FPS).

Todo se mide con una ventana de entrada de forma (1, seq_len, 3, H, W): lote 1, una ventana
por pasada, que es como funcionaría en una cámara.
"""
import copy
import io
import time

import numpy as np
import torch


def serialized_size_mb(m) -> float:
    """Tamaño en MB del state_dict serializado (igual para todas las variantes; no depende de que
    las clases estén importables al cargarlo, a diferencia de guardar el módulo completo)."""
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return buf.tell() / 1e6


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def measure_fps(m, dtype, device, seq_len, img_size, n_warmup=20, n_iters=50):
    """Fotogramas de entrada por segundo con lote 1: una ventana de seq_len fotogramas por pasada.

    Cada pasada cronometrada se sincroniza y se mide por separado, y se usa la MEDIANA
    (FPS = seq_len / mediana), para que unas pocas pasadas lentas por contención de la GPU o
    por subida del reloj no sesguen el resultado. Se excluyen las pasadas de calentamiento.
    """
    m.eval()
    x = torch.randn(1, seq_len, 3, *img_size, device=device)
    if dtype == 'fp16':
        x = x.half()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    times = []
    with torch.no_grad():
        for _ in range(n_warmup):
            m(x)
        _sync(device)
        for _ in range(n_iters):
            t0 = time.perf_counter()
            m(x)
            _sync(device)
            times.append(time.perf_counter() - t0)
    return seq_len / float(np.median(times))


def measure_fps_cpu(m, dtype, device, seq_len, img_size):
    """FPS forzado en CPU, para comparar todas las variantes en igualdad de condiciones
    (int8 solo puede correr en CPU, así que su FPS "normal" no es comparable con las filas de GPU).

    fp16 se omite (NaN): no hay kernels de media precisión utilizables en CPU. Las pasadas en
    CPU son lentas (ConvNeXt sobre todo), de ahí los pocos calentamientos/iteraciones.
    """
    if str(device) == 'cpu':
        return measure_fps(m, dtype, 'cpu', seq_len, img_size, n_warmup=2, n_iters=5)
    if dtype == 'fp16':
        return float('nan')
    try:
        cpu_model = copy.deepcopy(m).cpu()
        fps = measure_fps(cpu_model, dtype, 'cpu', seq_len, img_size, n_warmup=2, n_iters=5)
        del cpu_model
        return fps
    except Exception as exc:
        print(f'CPU FPS failed ({type(exc).__name__}: {exc}) — reporting NaN.')
        return float('nan')


def measure_gflops(m, device, seq_len, img_size):
    """GFLOPs de una ventana (lote 1), contados sobre el modelo fp32 denso. La poda no
    estructurada y el cambio de tipo no alteran la cantidad de operaciones densas, así que
    se reporta el mismo número para todas las variantes. Nota: el contador de PyTorch puede no
    contar la ruta fusionada del transformer temporal en modo eval; es poco frente al backbone."""
    from torch.utils.flop_counter import FlopCounterMode
    m.eval()
    x = torch.randn(1, seq_len, 3, *img_size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops() / 1e9
