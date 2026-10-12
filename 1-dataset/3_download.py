"""
3_download.py — PASO 3: descarga de videos de varias plataformas.

Descarga con yt-dlp todos los items 'approved' (marcados por 2_filter.py).
Soporta YouTube, TikTok, Instagram, Twitter/X, Facebook y URLs de noticias.

Los archivos quedan en --raw-dir como {país}/{item_id}.{ext} (por defecto
data/1_raw). Ese es exactamente el formato que lee el sitio de edición manual
(4_edit.py), que es el siguiente paso.

Cookies
-------
Facebook e Instagram exigen cookies de autenticación para descargar. Pasa
cookies.txt por plataforma con --facebook-cookies / --instagram-cookies /
--youtube-cookies / --tiktok-cookies / --twitter-cookies.

Requisito del sistema: las descargas de YouTube necesitan un runtime de
JavaScript (Deno) y el paquete yt-dlp-ejs; sin ellos yt-dlp responde "Requested
format is not available" para todos los videos. Ver README.md.

Uso
---
    python 3_download.py [--dry-run] [--batch-limit 50]
    python 3_download.py --source facebook --batch-limit 20
    python 3_download.py --country PE --classes Shooting
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 1-dataset/

from lib.common.classes import target_class_of
from lib.common.logger import get_logger
from lib.common.paths import RAW_DIR, from_portable, to_portable
from lib.scraping.downloader import PLATFORM_SLEEP, download_item
from lib.scraping.settings import (add_classes_arg, add_common_args, add_cookie_args,
                                   parse_classes_or_exit, resolve_cookies)
from lib.scraping.store import ItemStore


def main():
    parser = argparse.ArgumentParser(description="Descarga de videos de varias plataformas")
    add_common_args(parser)
    parser.add_argument("--batch-limit", type=int,   default=None,
                        help="Máximo de videos a descargar por corrida (por defecto: todos los aprobados)")
    parser.add_argument("--source",      default=None,
                        help="Descargar solo de esta plataforma (youtube, facebook, …)")
    parser.add_argument("--country",     default=None,
                        help="Descargar solo los items de este código de país (PE, BR, …)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Lista los items aprobados sin descargar")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Carpeta de salida de los videos descargados (por defecto: data/1_raw)")
    add_cookie_args(parser)
    add_classes_arg(parser,
                    "Solo descarga los items aprobados cuyo provisional_label (fijado por "
                    "2_filter.py) pertenece a alguna de estas clases.")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Intentos extra ante un DownloadError antes de rendirse con un "
                             "item (por defecto: 2); la resolución de formatos/JS de YouTube "
                             "falla a veces de forma intermitente aun en videos descargables")
    parser.add_argument("--retry-sleep", type=float, default=5.0,
                        help="Segundos de espera entre reintentos (por defecto: 5.0)")
    args = parser.parse_args()

    target_classes = parse_classes_or_exit(parser, args.classes)

    log, _  = get_logger("downloader", str(args.logs_dir))

    raw_dir = args.raw_dir
    cookies = resolve_cookies(args, log)
    cookies["news"] = None

    db      = ItemStore(args.items_csv)
    pending = db.get_by_status("approved", limit=None)

    if args.source:
        pending = [i for i in pending if (i.get("source") or "").lower() == args.source.lower()]
    if args.country:
        pending = [i for i in pending if (i.get("country") or "").upper() == args.country.upper()]
    if target_classes:
        pending = [i for i in pending if target_class_of(i.get("provisional_label")) in target_classes]
    if args.batch_limit:
        pending = pending[:args.batch_limit]

    log.info("=" * 60)
    log.info("[Downloader] Starting")
    log.info("  Raw dir     : %s", raw_dir)
    log.info("  Items CSV   : %s", args.items_csv)
    log.info("  Source      : %s", args.source or "all")
    log.info("  Country     : %s", args.country or "all")
    log.info("  Classes     : %s", ', '.join(sorted(target_classes)) if target_classes else "all (no filter)")
    log.info("  Batch limit : %s", args.batch_limit or "all")
    log.info("  Dry run     : %s", args.dry_run)
    log.info("  Cookies     : %s", {k: "✓" if v else "✗" for k, v in cookies.items()})
    log.info("=" * 60)
    log.info("[Downloader] %d items to process", len(pending))

    downloaded = failed = skipped = 0

    for i, item in enumerate(pending, 1):
        source  = (item.get("source") or "?").lower()
        country = (item.get("country") or "?").upper()
        label   = item.get("provisional_label") or "?"

        log.info("[%d/%d] %s  [%s|%s|%s]  %s",
                 i, len(pending), item["id"], source, country, label,
                 (item.get("raw_title") or "")[:60])

        # Si el archivo ya está en disco se omite (permite re-ejecutar sin duplicar)
        if item.get("download_path") and from_portable(item["download_path"]).exists():
            log.info("  → already on disk, skipping")
            skipped += 1
            continue

        if args.dry_run:
            log.info("  → [dry-run] would download")
            downloaded += 1
            continue

        updates = download_item(item, raw_dir, cookies, log,
                                max_retries=args.max_retries, retry_sleep=args.retry_sleep)

        if updates is None:
            db.update_item(item["id"], {"status": "failed"})
            log.info("  → FAILED")
            failed += 1
        else:
            if updates.get("download_path"):
                updates["download_path"] = to_portable(updates["download_path"])
            db.update_item(item["id"], updates)
            log.info("  → downloaded: %s", (updates.get("download_path") or "?")[:70])
            downloaded += 1

        time.sleep(PLATFORM_SLEEP.get(source, 2))

    log.info("=" * 60)
    log.info("[Downloader] %sDone", "Dry-run " if args.dry_run else "")
    log.info("  Downloaded : %d", downloaded)
    log.info("  Failed     : %d", failed)
    log.info("  Skipped    : %d  (already on disk)", skipped)
    if not args.dry_run:
        log.info("  DB summary : %s", db.status_summary())
    log.info("=" * 60)
    db.close()


if __name__ == "__main__":
    main()
