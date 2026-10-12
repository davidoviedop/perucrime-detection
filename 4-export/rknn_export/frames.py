"""
rknn_export/frames.py — fotogramas de calibración y de test, ya en el formato que recibe la NPU.

Preprocesado idéntico al del entrenamiento hasta el redimensionado: se decodifica con la misma
función del cargador de datos (lib.dataset.load_frames_fast) y se redimensiona con el mismo
`T.Resize(img_size)` de torchvision sobre PIL. La normalización NO se hace aquí: la aplica
RKNN (mean/std) o, en las referencias de PyTorch/ONNX, el script de puntuación.

Reutilización entre ventanas: las ventanas se solapan (abarcan 61 fotogramas pero avanzan de 16
en 16), así que un mismo fotograma aparece en varias. El preprocesado de test es determinista,
por lo que el embedding de un fotograma es el mismo en cualquier ventana: se calcula una sola vez
por fotograma único (en el test real, 3 408 fotogramas en vez de 10 848 pasadas por el backbone).
"""
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm

from lib.dataset import _list_videos, index_video, load_frames_fast


def window_records(data_root, sources, classes, split, cfg):
    """Registros (ruta, inicio, etiqueta) de todas las ventanas de un split, en el mismo orden
    que lib.dataset.build_loaders."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    recs = []
    for cls in classes:
        for src in sources:
            for vp in _list_videos(Path(data_root) / src / split / cls):
                recs.extend(index_video(vp, cls_to_idx[cls], cfg.seq_len, cfg.stride,
                                        frame_stride=cfg.frame_stride,
                                        max_windows=cfg.max_windows_per_video))
    return recs


def _resizer(img_size):
    resize = T.Resize(tuple(img_size))      # exactamente la primera transformación del entrenamiento

    def apply(frame_rgb: np.ndarray) -> np.ndarray:
        return np.asarray(resize(Image.fromarray(frame_rgb)), dtype=np.uint8)
    return apply


def build_frame_cache(recs, cfg):
    """Devuelve (frames, window_index):
      frames        uint8 (N, H, W, 3) RGB, un fotograma por cada (video, nº de fotograma) único
      window_index  int32 (n_ventanas, seq_len): fila de `frames` de cada fotograma de cada ventana
    """
    resize = _resizer(cfg.img_size)
    where: dict[tuple, int] = {}
    rows: list[np.ndarray] = []
    window_index = np.zeros((len(recs), cfg.seq_len), dtype=np.int32)

    for w, (vpath, start, _label) in enumerate(tqdm(recs, desc="test frames", unit="ventana")):
        frames = load_frames_fast(vpath, start, cfg.seq_len, tuple(cfg.img_size), cfg.frame_stride)
        for j, frame in enumerate(frames):
            key = (vpath, start + j * cfg.frame_stride)
            if key not in where:
                where[key] = len(rows)
                rows.append(resize(frame))
            window_index[w, j] = where[key]
    return np.stack(rows), window_index


def calibration_images(train_recs, cfg, per_class, seed):
    """Imágenes (uint8 RGB, ya redimensionadas) para calibrar la cuantización: `per_class`
    ventanas de ENTRENAMIENTO por clase, un fotograma al azar de cada una, mezcladas."""
    rng = random.Random(seed)
    resize = _resizer(cfg.img_size)
    by_class = defaultdict(list)
    for rec in train_recs:
        by_class[rec[2]].append(rec)

    picks = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        picks.extend(rng.sample(pool, min(per_class, len(pool))))
    rng.shuffle(picks)                      # así las primeras N líneas del dataset siguen mezclando clases

    images = []
    for vpath, start, _label in tqdm(picks, desc="calibración", unit="img"):
        frames = load_frames_fast(vpath, start, cfg.seq_len, tuple(cfg.img_size), cfg.frame_stride)
        images.append(resize(frames[rng.randrange(cfg.seq_len)]))
    return images


def save_calibration(images, out_dir):
    """Guarda las imágenes como PNG (sin pérdida) y escribe dataset.txt, un camino absoluto por línea,
    que es el formato del argumento `dataset` de rknn.build()."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = out_dir / f"calib_{i:04d}.png"
        Image.fromarray(img).save(p)
        paths.append(str(p.resolve()))
    dataset_txt = out_dir.parent / "dataset.txt"
    dataset_txt.write_text("\n".join(paths) + "\n")
    return dataset_txt
