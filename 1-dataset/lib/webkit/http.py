"""
Utilidades HTTP de los sitios de revisión: streaming de video con peticiones
Range (necesario para que el navegador pueda saltar a cualquier punto) y
validación de peticiones con errores en JSON.
"""
import mimetypes
import re
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from .media import video_dimensions


# ── Streaming de video ──────────────────────────────────────────────────────

def serve_video_file(path: Path):
    """Sirve un archivo de video respetando las cabeceras HTTP Range para poder buscar."""
    mime      = mimetypes.guess_type(str(path))[0] or "video/mp4"
    file_size = path.stat().st_size
    range_hdr = request.headers.get("Range")

    if not range_hdr:
        return send_file(path, mimetype=mime)

    match = re.search(r"bytes=(\d+)-(\d*)", range_hdr)
    if not match:
        return Response(status=416)

    byte1  = int(match.group(1))
    byte2  = int(match.group(2)) if match.group(2) else file_size - 1
    byte2  = min(byte2, file_size - 1)
    length = byte2 - byte1 + 1

    with open(path, "rb") as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype=mime)
    rv.headers["Content-Range"]  = f"bytes {byte1}-{byte2}/{file_size}"
    rv.headers["Accept-Ranges"]  = "bytes"
    rv.headers["Content-Length"] = str(length)
    return rv


# ── Validación de peticiones ────────────────────────────────────────────────

class ApiError(Exception):
    """La lanzan los validadores de peticiones; se responde como {"error": mensaje} con `status`."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def install_error_handler(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def _handle(err: ApiError):
        return jsonify({"error": err.message}), err.status


def require_class(data: dict, valid_classes: list[str]) -> str:
    """El "class" del cuerpo de la petición, que debe ser una de `valid_classes`."""
    new_class = (data.get("class") or "").strip()
    if new_class not in valid_classes:
        raise ApiError(f"unknown class {new_class!r}")
    return new_class


def parse_trim(trim_in, trim_out, duration: float, bounds_msg: str) -> tuple[float, float]:
    """Valida una ventana de recorte contra la duración del video y la acota a [0, duración]."""
    try:
        trim_in, trim_out = float(trim_in), float(trim_out)
    except (TypeError, ValueError):
        raise ApiError("trim_in/trim_out must be numbers") from None
    if not (-1e-6 <= trim_in < trim_out <= duration + 1e-6):
        raise ApiError(bounds_msg)
    return max(0.0, trim_in), min(duration, trim_out)


def parse_crop(crop: dict, video_path: Path, what: str) -> dict:
    """Valida una caja absoluta {left, top, right, bottom} contra el tamaño real de fotograma del video."""
    try:
        left, top     = int(crop["left"]),  int(crop["top"])
        right, bottom = int(crop["right"]), int(crop["bottom"])
    except (KeyError, TypeError, ValueError):
        raise ApiError("invalid crop box") from None
    if not (0 <= left < right and 0 <= top < bottom):
        raise ApiError("invalid crop box")
    vid_w, vid_h = video_dimensions(video_path)
    if vid_w and vid_h and (right > vid_w or bottom > vid_h):
        raise ApiError(f"crop box {right}x{bottom} exceeds the {what}'s actual "
                       f"frame size {vid_w}x{vid_h}")
    return {"left": left, "top": top, "right": right, "bottom": bottom}
