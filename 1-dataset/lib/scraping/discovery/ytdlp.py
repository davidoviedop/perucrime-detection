"""
Descubrimiento con yt-dlp: búsquedas de YouTube y extracción plana de URLs de
canales / cuentas / páginas (canales de YouTube, TikTok, Instagram, Twitter/X).
No se descarga nada: solo se recolecta la lista de URLs de video y sus títulos.
"""
import yt_dlp


def _flat_opts(limit: int) -> dict:
    return {
        "quiet":        True,
        "no_warnings":  True,
        "extract_flat": True,
        "playlistend":  limit,
    }


def search_youtube(query: str, max_results: int, log) -> tuple[list[dict], bool]:
    """Extracción plana de ytsearch; devuelve (entradas, hubo_error)."""
    try:
        with yt_dlp.YoutubeDL(_flat_opts(max_results)) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        return info.get("entries") or [], False
    except Exception as exc:
        log.warning(f"    [!] yt-dlp error for query '{query}': {exc}")
        return [], True


def scrape_page(url: str, max_results: int, log) -> tuple[list[dict], bool]:
    """Extracción plana de la URL de un canal/página/cuenta; devuelve (entradas, hubo_error).
    hubo_error distingue "yt-dlp lanzó una excepción" de "realmente 0 videos",
    para que quien llame detecte que una plataforma/extractor completo está caído
    (p. ej. yt-dlp lo marcó como roto) y no que una sola cuenta esté vacía o
    haya cambiado de nombre."""
    try:
        with yt_dlp.YoutubeDL(_flat_opts(max_results)) as ydl:
            info = ydl.extract_info(url, download=False)
        if info is None:
            return [], False
        # Algunos extractores devuelven una playlist plana directo; otros la envuelven en entries
        entries = info.get("entries") or []
        # Desenvuelve un nivel extra de anidación (p. ej. cuenta de TikTok → pestaña de videos)
        if entries and isinstance(entries[0], dict) and "entries" in entries[0]:
            entries = entries[0].get("entries") or []

        # Cuando yt-dlp no tiene extractor propio para un sitio (o está roto;
        # p. ej. x.com/Twitter hoy no tiene), recurre al raspador HTML
        # genérico, que puede inventar "entradas" que solo apuntan de vuelta a
        # la página del perfil en vez de a un video real (webpage_url == la
        # URL de entrada, sin ie_key/url reales). No son videos descubiertos:
        # se filtran antes de que quien llame las convierta en items falsos.
        real_entries = [
            e for e in entries
            if isinstance(e, dict) and (e.get("webpage_url") or e.get("url")) != url
        ]
        if len(real_entries) < len(entries):
            log.debug(f"    [!] dropped {len(entries) - len(real_entries)} placeholder "
                     f"entr{'y' if len(entries) - len(real_entries) == 1 else 'ies'} "
                     f"pointing back at '{url}' itself (no dedicated extractor for this site?)")
        return real_entries, False
    except Exception as exc:
        log.warning(f"    [!] yt-dlp error for page '{url}': {exc}")
        return [], True
