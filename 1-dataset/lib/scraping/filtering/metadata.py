"""
Obtención de solo metadatos de una URL descubierta (sin descargar el video):
título, descripción, autor, fecha de subida, duración y vistas, vía yt-dlp.
"""
import yt_dlp


def fetch_metadata(url: str, cookies_file: str | None = None) -> tuple[dict | None, str | None]:
    """Devuelve (metadatos, None) si tuvo éxito, o (None, fragmento_del_error) si falló.
    El error se expone (no se traga) para que quien llame distinga una
    restricción de edad o un muro de login de una URL realmente muerta."""
    opts: dict = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        # Aquí solo queremos metadatos (título/descripción/duración/vistas),
        # nunca un stream descargable; pero yt-dlp ejecuta igual su selector
        # de formato aun con skip_download=True y lanza "Requested format is
        # not available" si ningún formato coincide (videos con restricción de
        # edad o de región sin autenticación completa). Los metadatos se pueden
        # obtener igual: se omite la comprobación de formato en vez de perder
        # el item.
        "ignore_no_formats_error": True,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if info is None:
            return None, "no_info_returned"
        return {
            "raw_title":       (info.get("title") or "").strip(),
            "raw_description": (info.get("description") or "").strip()[:4000],
            "channel":         info.get("channel") or info.get("uploader") or "",
            "upload_date":     info.get("upload_date") or "",
            "duration":        info.get("duration"),
            "view_count":      info.get("view_count"),
        }, None
    except Exception as exc:
        return None, str(exc)
