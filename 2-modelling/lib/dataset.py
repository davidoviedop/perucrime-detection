"""
lib/dataset.py — dataset de ventanas deslizantes sobre los videos del dataset final.

Estructura de carpetas: data_root/{fuente}/{split}/{clase}/*.mp4
Es lo que produce 8_export.py de la etapa 1 (data/3_dataset). Cada fuente (p. ej.
"scrapped", "ucf-crime", "vioperu") aporta su propio árbol train/val/test/clase; los
videos de todas las fuentes se juntan por split. Una fuente sin datos aporta cero
videos.

Cada clip ya es un segmento de una sola clase, recortado y de una sola escena (se
cortó en la etapa 1), así que no hace falta detección de escenas ni metadatos por
segmento: cada clip se indexa como un único rango de ventanas.

Una ventana son `seq_len` fotogramas tomados cada `frame_stride` fotogramas fuente, así
que abarca (seq_len - 1) * frame_stride + 1 fotogramas del video. Los inicios de
ventana avanzan de a `stride` fotogramas.
"""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm


# ── Transformaciones ──────────────────────────────────────────────────────────

class WindowTrainTransform:
    """Aumentación de entrenamiento por ventana.
    Recibe una lista de imágenes PIL y devuelve un tensor (T, 3, H, W).
    """
    per_window = True

    def __init__(self, img_size, flip_p=0.5,
                 brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05):
        self.resize = T.Resize(img_size)
        self.flip_p = flip_p
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast,
                                    saturation=saturation, hue=hue)

    def __call__(self, images):
        # Un solo sorteo por ventana (torch.rand / get_params usan el generador de torch,
        # que el DataLoader re-siembra por worker y por época).
        flip = torch.rand(1).item() < self.flip_p
        order, b, c, s, h = T.ColorJitter.get_params(
            self.jitter.brightness, self.jitter.contrast,
            self.jitter.saturation, self.jitter.hue)

        out = []
        for img in images:
            img = self.resize(img)
            if flip:
                img = TF.hflip(img)
            for fn_id in order:   # mismo orden aleatorio de operaciones que T.ColorJitter
                if fn_id == 0 and b is not None:
                    img = TF.adjust_brightness(img, b)
                elif fn_id == 1 and c is not None:
                    img = TF.adjust_contrast(img, c)
                elif fn_id == 2 and s is not None:
                    img = TF.adjust_saturation(img, s)
                elif fn_id == 3 and h is not None:
                    img = TF.adjust_hue(img, h)
            out.append(TF.normalize(TF.to_tensor(img), _MEAN, _STD))
        return torch.stack(out)


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def make_transforms(img_size):
    """Devuelve (transformación_de_entrenamiento, transformación_de_validación) para el tamaño dado."""
    train = WindowTrainTransform(img_size)
    val = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])
    return train, val


# ── Descubrimiento de fuentes y clases ────────────────────────────────────────

def list_sources(data_root):
    """Subcarpetas de data_root, ordenadas: cada una es un árbol de fuente independiente."""
    data_root = Path(data_root)
    return sorted(p.name for p in data_root.iterdir() if p.is_dir())


def discover_classes(data_root, sources, split='train'):
    """Unión de los nombres de carpeta de clase en el `split` de todas las fuentes."""
    data_root = Path(data_root)
    classes = set()
    for src in sources:
        split_dir = data_root / src / split
        if split_dir.exists():
            classes.update(p.name for p in split_dir.iterdir() if p.is_dir())
    return sorted(classes)


def _list_videos(folder):
    """Archivos .mp4 y .avi de la carpeta, ordenados (lista vacía si no existe)."""
    if not folder.exists():
        return []
    return sorted(folder.glob('*.mp4')) + sorted(folder.glob('*.avi'))


# ── Indexado de ventanas ──────────────────────────────────────────────────────

def frame_count(vpath):
    """Cantidad de fotogramas del video según su cabecera (sin decodificarlo)."""
    cap = cv2.VideoCapture(str(vpath))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

def index_video(vpath, label, seq_len, stride,
                frame_stride=1, max_windows=None):
    """Desliza ventanas de seq_len fotogramas por todo el clip.

    Los clips ya son de una sola clase y un solo segmento, así que el clip completo
    se trata como un único rango de ventanas.

    Args:
        vpath:        ruta del video
        label:        índice entero de la clase
        seq_len:      fotogramas muestreados por ventana
        stride:       paso en fotogramas entre inicios de ventana consecutivos
                      (controla el solapamiento entre ventanas, no entre fotogramas)
        frame_stride: paso entre fotogramas muestreados dentro de una ventana.
                      frame_stride=1 → fotogramas consecutivos
                      frame_stride=4 → un fotograma de cada 4; la ventana abarca
                                       (seq_len-1)*frame_stride+1 fotogramas fuente
        max_windows:  si se indica, reduce las ventanas a esa cantidad tomándolas
                      a intervalos regulares

    Devuelve:
        lista de tuplas (vpath_str, start_frame, label)
    """
    n = frame_count(vpath)

    # fotogramas fuente que consume realmente una ventana
    span = (seq_len - 1) * frame_stride + 1

    if n < span:
        return []

    recs = [(str(vpath), s, label)
            for s in range(0, n - span + 1, stride)]

    if max_windows and len(recs) > max_windows:
        step = len(recs) / max_windows
        recs = [recs[int(i * step)] for i in range(max_windows)]

    return recs


# ── Carga de fotogramas ───────────────────────────────────────────────────────

def load_frames_fast(vpath, start, n, img_size, frame_stride=1):
    """Lee n fotogramas de vpath desde el índice `start`, avanzando frame_stride entre muestras.

    Estrategia: un solo salto a `start`, luego se leen `span` fotogramas de forma
    secuencial y se submuestrean. La decodificación secuencial es mucho más rápida que
    llamar a cap.set() varias veces en H.264, porque cada salto explícito debe decodificar
    desde el keyframe más cercano.

    Si el video termina antes, rellena con el último fotograma válido (o uno negro).

    Args:
        vpath:        ruta del video
        start:        índice del primer fotograma a leer
        n:            fotogramas a devolver (= seq_len)
        img_size:     tupla (H, W), solo para el fotograma negro de relleno
        frame_stride: paso entre fotogramas muestreados en el video fuente

    Devuelve:
        lista de n arreglos numpy RGB uint8, cada uno de forma (H, W, 3)
    """
    span = (n - 1) * frame_stride + 1   # fotogramas fuente a leer secuencialmente

    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    raw = []
    for _ in range(span):
        ret, frm = cap.read()
        if not ret:
            break
        raw.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    cap.release()

    # submuestreo: un fotograma de cada frame_stride
    frames = raw[::frame_stride][:n]

    # relleno si el video terminó antes
    while len(frames) < n:
        frames.append(
            frames[-1] if frames
            else np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        )

    return frames


# ── Dataset ───────────────────────────────────────────────────────────────────

class VideoWindowDataset(Dataset):
    """Dataset perezoso de ventanas deslizantes.

    Registros: lista de tuplas (vpath_str, start_frame, label).
    Los fotogramas se decodifican del disco solo cuando se llama a __getitem__.

    Args:
        records:      lista de (vpath_str, start_frame, label)
        transform:    transformación de torchvision aplicada a cada fotograma
        seq_len:      fotogramas por ventana
        img_size:     (H, W), para el fotograma negro de relleno de load_frames_fast
        frame_stride: paso entre fotogramas muestreados dentro de una ventana (por defecto 1)
        temporal_jitter: jitter aleatorio en fotogramas fuente al muestrear la ventana (0 = sin jitter)
        frame_counts: dict {vpath: n_fotogramas} para calcular el jitter temporal
    """
    def __init__(self, records, transform, seq_len, img_size, frame_stride=1,
                 temporal_jitter=0, frame_counts=None):
        self.records      = records
        self.transform    = transform
        self.seq_len      = seq_len
        self.img_size     = img_size
        self.frame_stride = frame_stride
        self.temporal_jitter = int(temporal_jitter)
        self.frame_counts    = frame_counts or {}
        if self.temporal_jitter < 0:
            raise ValueError('temporal_jitter debe ser >= 0')
        if self.temporal_jitter and not self.frame_counts:
            raise ValueError('temporal_jitter > 0 necesita frame_counts {vpath: n_fotogramas}')

    def __len__(self):
        return len(self.records)

    def _jittered_start(self, vpath, start):
        """Inicio desplazado al azar en [start, start + temporal_jitter], sin salir del clip.
        Se usa torch.randint: el DataLoader vuelve a sembrar el generador de torch en cada
        worker y en cada época, así que los desplazamientos cambian entre workers y épocas
        y siguen la semilla global de la corrida.
        """
        span      = (self.seq_len - 1) * self.frame_stride + 1
        max_start = self.frame_counts[vpath] - span      # último inicio que cabe entero
        if max_start <= start:                           # ventana pegada al final: no se mueve
            return start
        hi = min(start + self.temporal_jitter, max_start)
        return int(torch.randint(start, hi + 1, (1,)).item())    
    

    def __getitem__(self, idx):
        vpath, start, label = self.records[idx]
        if self.temporal_jitter:
            start = self._jittered_start(vpath, start)
        frames = load_frames_fast(
            vpath, start, self.seq_len, self.img_size, self.frame_stride
        )
        images = [Image.fromarray(f) for f in frames]
        if getattr(self.transform, 'per_window', False):
            seq = self.transform(images)            # mismo sorteo para toda la ventana
        else:
            seq = torch.stack([self.transform(im) for im in images])   # val/test: determinista
        return seq, torch.tensor(label, dtype=torch.long)



# ── Punto de entrada ──────────────────────────────────────────────────────────

def build_loaders(
    data_root, sources, classes, cls_to_idx,
    seq_len, stride, img_size, batch_size, num_workers,
    frame_stride=1,
    max_windows_per_video=None,
    temporal_jitter=0,
):
    """Construye los DataLoaders de train / val / test.

    Lee data_root/{fuente}/train|val|test/{clase}/ de cada fuente en `sources`,
    juntando los videos de todas las fuentes por split y clase. Usa los videos que haya,
    tal cual: no filtra ni vuelve a dividir.

    Args:
        data_root:             carpeta raíz con una subcarpeta por fuente
        sources:               lista de nombres de fuente (subcarpetas de data_root)
        classes:               lista ordenada de nombres de clase
        cls_to_idx:            dict nombre de clase → índice entero
        seq_len:               fotogramas muestreados por ventana
        stride:                paso entre inicios de ventana (fotogramas fuente)
        img_size:              tupla (H, W) para redimensionar
        batch_size:            tamaño de lote del DataLoader
        num_workers:           procesos de carga del DataLoader
        frame_stride:          paso entre fotogramas muestreados dentro de una ventana.
                               Cobertura temporal efectiva por ventana:
                               (seq_len - 1) * frame_stride + 1 fotogramas fuente.
                               A 25 FPS, frame_stride=4 → ~160 ms entre muestras.
        max_windows_per_video: tope de ventanas por video (None = sin tope)
        temporal_jitter:       desplazamiento aleatorio máximo (fotogramas fuente) del
                               inicio de cada ventana de train; 0 = sin aumentación temporal        

    Devuelve:
        train_loader, val_loader, test_loader,
        tr_recs, vl_recs, test_recs   — registros de ventanas (val/test se necesitan
                                         para agrupar las predicciones por video)
    """
    data_root = Path(data_root)
    train_tf, val_tf = make_transforms(img_size)

    # ── Resumen de la configuración ───────────────────────────────────────────
    span_frames  = (seq_len - 1) * frame_stride + 1
    span_seconds = span_frames / 25.0   # supone 25 FPS
    ms_between   = frame_stride * (1000 / 25)

    print(f'Sources      : {sources}')
    print(f'seq_len      : {seq_len} frames')
    print(f'frame_stride : {frame_stride}  ({ms_between:.0f} ms between samples)')
    print(f'window span  : {span_frames} source frames  ({span_seconds:.2f} s at 25 FPS)')
    print(f'win stride   : {stride} source frames between window starts')
    print(f'temp. jitter : {temporal_jitter} source frames (train only)')
    print()

    # ── Cantidad de videos desde el disco, juntando las fuentes ───────────────
    print('Video counts (from disk, pooled across sources):')
    print(f'  {"Class":<20}  {"train":>5}  {"val":>5}  {"test":>5}  {"total":>6}')
    print('  ' + '-' * 48)
    for cls in classes:
        counts = [
            sum(len(_list_videos(data_root / src / s / cls)) for src in sources)
            for s in ('train', 'val', 'test')
        ]
        print(f'  {cls:<20}  {counts[0]:>5}  {counts[1]:>5}  {counts[2]:>5}'
              f'  {sum(counts):>6}')
    print()

    # ── Indexado de ventanas por split ────────────────────────────────────────
    split_recs = {}
    for split in ('train', 'val', 'test'):
        all_tasks = [
            (cls, vp)
            for cls in classes
            for src in sources
            for vp in _list_videos(data_root / src / split / cls)
        ]
        cls_stats = {cls: [0, 0] for cls in classes}
        recs = []

        for cls, vp in tqdm(all_tasks, desc=f'{split:5s} indexing', unit='vid'):
            wins = index_video(
                vp, cls_to_idx[cls], seq_len, stride,
                frame_stride=frame_stride,
                max_windows=max_windows_per_video,
            )
            recs.extend(wins)
            cls_stats[cls][0] += 1
            cls_stats[cls][1] += len(wins)

        print(f'  {"Class":<20}  {"videos":>6}  {"windows":>7}')
        for cls in classes:
            v, w = cls_stats[cls]
            print(f'  {cls:<20}  {v:>6}  {w:>7}')
        print()
        split_recs[split] = recs

    tr_recs, vl_recs, test_recs = (
        split_recs['train'], split_recs['val'], split_recs['test']
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
        # El desplazamiento necesita saber dónde termina cada clip de train.
    tr_counts = ({vp: frame_count(vp) for vp in sorted({r[0] for r in tr_recs})}
                 if temporal_jitter else None)
    train_ds = VideoWindowDataset(
        tr_recs,   train_tf, seq_len, img_size, frame_stride,
        temporal_jitter=temporal_jitter, frame_counts=tr_counts)
    val_ds   = VideoWindowDataset(
        vl_recs,   val_tf,   seq_len, img_size, frame_stride)
    test_ds  = VideoWindowDataset(
        test_recs, val_tf,   seq_len, img_size, frame_stride)

    # Muestreador de frecuencia inversa: lotes de entrenamiento balanceados por clase
    cls_counts     = Counter(r[2] for r in tr_recs)
    sample_weights = [1.0 / max(cls_counts[r[2]], 1) for r in tr_recs]
    sampler        = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    print('Class sampling weights:')
    for i, cls in enumerate(classes):
        print(f'  {cls:<20}  n={cls_counts[i]:6d}  w={1.0/max(cls_counts[i],1):.6f}')
    print()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    xb, yb = next(iter(train_loader))
    print(f'Batch shape : {tuple(xb.shape)}')
    print(f'Labels (8)  : {yb.tolist()[:8]}')

    return train_loader, val_loader, test_loader, tr_recs, vl_recs, test_recs
