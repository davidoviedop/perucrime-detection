"""
1_discover.py — PASO 1: descubrimiento de videos en varias plataformas.

Descubre URLs de video y las inserta como items con status='discovered' en
data/metadata/items.csv (el estado del scraping, ver --items-csv). Aquí no se descarga nada:
eso lo hace 3_download.py, después de que 2_filter.py apruebe los items.

Estrategias
-----------
  youtube   — ytsearch de yt-dlp por consulta + listados de canales (sources.yaml)
  facebook  — raspador con requests (y Playwright de respaldo) del listado /videos/
              de cada página (cookies.txt opcional con --cookies)
  tiktok    — extracción plana con yt-dlp de las cuentas configuradas
  instagram — extracción plana con yt-dlp de las cuentas configuradas
  twitter   — extracción plana con yt-dlp de las cuentas configuradas

Uso
---
    python 1_discover.py --source youtube [--countries PE,CO] [--dry-run]
    python 1_discover.py --source facebook [--countries PE] --cookies fb.txt
    python 1_discover.py --source youtube --classes Shooting,Assault

Cookies de Facebook (opcional, mejora el rendimiento): expórtalas desde
Chrome/Firefox con la extensión "Get cookies.txt LOCALLY" y pasa el archivo con
--cookies.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 1-dataset/

import yaml

from lib.common.logger import get_logger
from lib.common.paths import DATASET_DIR
from lib.scraping.discovery.constants import COUNTRY_KEY_MAP, SUPPORTED_SOURCES
from lib.scraping.discovery.runners import run_page_based, run_youtube
from lib.scraping.settings import add_classes_arg, add_common_args, parse_classes_or_exit
from lib.scraping.store import ItemStore


def main():
    parser = argparse.ArgumentParser(description="Descubrimiento de videos en varias plataformas")
    parser.add_argument("--source", default="youtube", choices=SUPPORTED_SOURCES,
                        help="Plataforma a raspar (por defecto: youtube)")
    parser.add_argument("--countries", default=None,
                        help="Códigos de país ISO-2 separados por comas, p. ej. PE,CO "
                             "(por defecto: todos los configurados)")
    parser.add_argument("--max-results", type=int, default=25,
                        help="Máximo de resultados por consulta/página (por defecto: 25)")
    parser.add_argument("--cookies", default=None, metavar="ARCHIVO",
                        help="cookies.txt (Netscape) para autenticarse en Facebook")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula la corrida sin escribir en la BD")
    parser.add_argument("--sources-config", type=Path, default=DATASET_DIR / "config" / "sources.yaml",
                        help="Configuración de descubrimiento por plataforma (por defecto: config/sources.yaml)")
    add_common_args(parser)
    add_classes_arg(parser,
                    "Solo afecta a --source youtube: omite las consultas de búsqueda que se "
                    "clasifican en otra clase. Los canales/páginas/cuentas de cualquier plataforma "
                    "no se pueden preclasificar antes de descubrirlos: fíltralos en 2_filter.py, "
                    "cuando cada item ya tiene título/descripción para clasificar.")
    args = parser.parse_args()

    target_classes = parse_classes_or_exit(parser, args.classes)

    sources = yaml.safe_load(args.sources_config.read_text())
    log, _  = get_logger("scrapper", str(args.logs_dir))

    source      = args.source
    max_results = args.max_results
    fb_cookies  = args.cookies

    if args.countries:
        countries = [c.strip().upper() for c in args.countries.split(",")]
    else:
        platform_cfg = sources.get(source, {})
        countries = [COUNTRY_KEY_MAP[k] for k in platform_cfg if k in COUNTRY_KEY_MAP]

    log.info("=" * 60)
    log.info(f"[Scrapper] Starting  source={source}")
    log.info(f"  Countries:   {', '.join(countries) or 'none configured'}")
    log.info(f"  Max results: {max_results} per query/page")
    log.info(f"  Items CSV:   {args.items_csv}")
    log.info(f"  Dry run:     {args.dry_run}")
    log.info(f"  Classes:     {', '.join(sorted(target_classes)) if target_classes else 'all (no filter)'}")
    if target_classes and source != "youtube":
        log.info("  [!] --classes only pre-filters youtube search queries — "
                 f"'{source}' discovers everything; filter by class in 2_filter.py instead.")
    if source == "facebook":
        log.info(f"  FB cookies:  {fb_cookies or 'none (public pages only)'}")
    log.info("=" * 60)

    store = ItemStore(args.items_csv)

    if source == "youtube":
        inserted, skipped = run_youtube(sources, countries, max_results, store, args.dry_run, log,
                                        target_classes=target_classes)
    elif source == "facebook":
        inserted, skipped = run_page_based(
            source, sources, countries, max_results, store, args.dry_run, log,
            fb_cookies=fb_cookies,
        )
    else:
        inserted, skipped = run_page_based(source, sources, countries, max_results, store, args.dry_run, log)

    log.info("\n" + "=" * 60)
    log.info(f"[Scrapper] {'Dry-run ' if args.dry_run else ''}Done")
    if args.dry_run:
        log.info(f"  Would insert: {inserted}")
    else:
        log.info(f"  Inserted:     {inserted}")
        log.info(f"  Skipped (dup): {skipped}")
        log.info(f"  DB totals:    {store.status_summary()}")
    log.info("=" * 60)
    store.close()


if __name__ == "__main__":
    main()
