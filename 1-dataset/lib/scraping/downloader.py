"""
Descarga de un solo video con yt-dlp: formato según la plataforma, pausas
anti-límite de tasa y reintentos ante DownloadError.
"""
import time
from pathlib import Path

import yt_dlp


# Segundos de pausa entre descargas, por plataforma de origen.
# YouTube es permisivo; TikTok/Instagram limitan la tasa agresivamente.
PLATFORM_SLEEP = {
    "youtube":   1,
    "facebook":  3,
    "instagram": 4,
    "tiktok":    3,
    "twitter":   2,
    "news":      1,
}

# Especificación de formato por plataforma.
# YouTube ofrece video y audio por separado; la mayoría de las demás los sirve juntos.
PLATFORM_FORMAT = {
    "youtube":   "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
    "facebook":  "best[height<=720]/best",
    "instagram": "best[height<=720]/best",
    "tiktok":    "best",
    "twitter":   "best[height<=720]/best",
    "news":      "best",
}
DEFAULT_FORMAT = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best"


def build_ydl_opts(out_path: str, format_spec: str, cookies_file: str | None) -> dict:
    opts = {
        "format":              format_spec,
        "merge_output_format": "mp4",
        "outtmpl":             out_path,
        "quiet":               True,
        "no_warnings":         True,
        "ignoreerrors":        False,
        "retries":             3,
        "fragment_retries":    3,
        "sleep_interval":      1,
        "max_sleep_interval":  3,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    return opts


def download_item(
    item: dict,
    raw_dir: Path,
    cookies: dict[str, str | None],
    log,
    max_retries: int = 2,
    retry_sleep: float = 5.0,
) -> dict | None:
    """
    Descarga un video con yt-dlp.
    Devuelve un dict con las actualizaciones para la BD si tuvo éxito, o None si falló.

    Reintenta ante DownloadError (no ante otras excepciones) hasta max_retries
    intentos extra. La resolución del desafío JS / PO-token de YouTube (necesaria
    para elegir el formato; requiere Deno instalado) a veces falla de forma
    intermitente aun en un video perfectamente descargable: la misma URL que
    lanza "Requested format is not available" en un intento puede funcionar en
    un reintento idéntico segundos después, sin cambiar código ni configuración.
    Un único fallo no es evidencia fiable de que el video no se pueda descargar.
    """
    source  = (item.get("source") or "unknown").lower()
    country = (item.get("country") or "XX").upper()
    url     = item.get("url", "")

    out_dir = raw_dir / country
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{item['id']}.%(ext)s")

    fmt          = PLATFORM_FORMAT.get(source, DEFAULT_FORMAT)
    cookies_file = cookies.get(source)

    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            with yt_dlp.YoutubeDL(build_ydl_opts(out_path, fmt, cookies_file)) as ydl:
                info = ydl.extract_info(url, download=True)

            if info is None:
                log.warning("  [!] yt-dlp returned None for URL: %s", url[:80])
                return None

            # Busca el archivo escrito (la extensión puede variar)
            matches = sorted(out_dir.glob(f"{item['id']}.*"))
            download_path = str(matches[0]) if matches else None

            return {
                "status":          "downloaded",
                "download_path":   download_path,
                "upload_date":     info.get("upload_date") or "",
                "raw_description": (info.get("description") or "")[:4000],
                "channel":         info.get("channel") or info.get("uploader") or item.get("channel", ""),
            }

        except yt_dlp.utils.DownloadError as exc:
            if attempt < attempts:
                log.warning("  [!] DownloadError (attempt %d/%d): %s — retrying in %.0fs",
                           attempt, attempts, str(exc)[:120], retry_sleep)
                time.sleep(retry_sleep)
                continue
            log.warning("  [!] DownloadError (attempt %d/%d, giving up): %s",
                       attempt, attempts, str(exc)[:120])
            return None
        except Exception as exc:
            log.warning("  [!] Unexpected error: %s", str(exc)[:120])
            return None

    return None
