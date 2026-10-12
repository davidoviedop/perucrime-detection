"""Convierte una entrada cruda de yt-dlp / del raspador en una fila de la tabla items de la BD de staging."""
import hashlib
from datetime import datetime, timezone


def make_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def entry_to_item(entry: dict, source: str, country_code: str) -> dict | None:
    """Convierte una entrada cruda de yt-dlp en un dict de item del pipeline (None si no tiene una URL utilizable)."""
    url = (
        entry.get("webpage_url")
        or entry.get("url")
        or (f"https://www.youtube.com/watch?v={entry['id']}" if source == "youtube" and entry.get("id") else None)
    )
    if not url or not url.startswith("http"):
        return None

    now = datetime.now(timezone.utc).isoformat()
    return {
        "id":          make_id(url),
        "url":         url,
        "source":      source,
        "country":     country_code,
        "raw_title":   entry.get("title") or "",
        "channel":     entry.get("channel") or entry.get("uploader") or "",
        "status":      "discovered",
        "created_at":  now,
        "updated_at":  now,
    }
