"""
Wrappers de ffmpeg / ffprobe (duración y tamaño del video, y extracción de un
clip con recorte temporal y espacial en una sola pasada) y las convenciones de
nombres de archivo en las que se apoyan los sitios de revisión.

Todo se recodifica con libx264 (CRF 18) y se descarta el audio: los modelos
solo usan los fotogramas.
"""
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Los nombres de clips y de videos crudos empiezan con un id de item de ancho
# fijo (sha256(url)[:16]; ver lib/scraping/discovery/items.py), así que el item
# al que pertenece un archivo se lee de su nombre sin ningún almacén de metadatos.
ITEM_ID_LEN = 16

_PROBE_TIMEOUT  = 15    # segundos: ffprobe debería ser casi instantáneo
_ENCODE_TIMEOUT = 300   # segundos: los videos crudos pueden durar varios minutos


def video_duration(path: Path) -> float:
    """Duración en segundos, o 0.0 si no se puede leer."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def video_dimensions(path: Path) -> tuple[int, int]:
    """(ancho, alto) en píxeles, o (0, 0) si no se pueden leer."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return 0, 0
    w = h = 0
    for line in result.stdout.splitlines():
        if line.startswith("width="):
            try: w = int(line.split("=")[1])
            except ValueError: pass
        elif line.startswith("height="):
            try: h = int(line.split("=")[1])
            except ValueError: pass
    return w, h


def _run_ffmpeg(args: list[str], tmp: Path) -> bool:
    """Ejecuta ffmpeg con timeout (para que una petición/hilo no se cuelgue para
    siempre) y siempre borra el archivo temporal de salida si algo falla."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=_ENCODE_TIMEOUT)
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink()
        return False
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        return False
    return True


def _crop_filter(box: dict) -> str:
    """Filtro crop de ffmpeg para una caja absoluta left/top/right/bottom. Ancho y
    alto se redondean hacia abajo a números pares (libx264 lo exige)."""
    w = (int(box["right"])  - int(box["left"])) & ~1
    h = (int(box["bottom"]) - int(box["top"]))  & ~1
    return f"crop={w}:{h}:{int(box['left'])}:{int(box['top'])}"


_ENCODE_ARGS = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", "-loglevel", "error"]


def extract_clip(src: Path, out_path: Path, trim_in: float, trim_out: float, crop: dict | None) -> bool:
    """Corta [trim_in, trim_out] de `src` (opcionalmente recortado) hacia un
    `out_path` nuevo; `src` nunca se abre para escritura. El recorte temporal y
    el espacial comparten una sola pasada de recodificación."""
    args = ["ffmpeg", "-y", "-i", str(src), "-ss", f"{trim_in:.3f}", "-to", f"{trim_out:.3f}"]
    if crop:
        args += ["-vf", _crop_filter(crop)]
    args += [*_ENCODE_ARGS, str(out_path)]
    return _run_ffmpeg(args, out_path)
