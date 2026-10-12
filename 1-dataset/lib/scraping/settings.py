"""
lib/scraping/settings.py — utilidades compartidas por los tres scripts de scraping
(1_discover.py, 2_filter.py, 3_download.py): rutas por defecto, las opciones de
cookies por plataforma y el filtro --classes.
"""
import argparse
from pathlib import Path

from lib.common.classes import TARGET_CLASSES, parse_classes
from lib.common.paths import DATASET_DIR, ITEMS_CSV

PLATFORMS = ("youtube", "facebook", "instagram", "tiktok", "twitter")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """--items-csv / --logs-dir, idénticos en todos los scripts."""
    parser.add_argument("--items-csv", type=Path, default=ITEMS_CSV,
                        help="CSV con el estado de los items descubiertos "
                             "(por defecto: data/metadata/items.csv; se crea si no existe)")
    parser.add_argument("--logs-dir", type=Path, default=DATASET_DIR / "data" / "logs",
                        help="Carpeta de los logs (por defecto: data/logs)")


def add_cookie_args(parser: argparse.ArgumentParser, platforms=PLATFORMS) -> None:
    """Opciones --<plataforma>-cookies (cookies.txt en formato Netscape)."""
    for platform in platforms:
        parser.add_argument(f"--{platform}-cookies", type=Path, default=None, metavar="ARCHIVO",
                            help=f"cookies.txt (Netscape) para {platform}")


def add_classes_arg(parser: argparse.ArgumentParser, what: str) -> None:
    parser.add_argument("--classes", default=None,
                        help="Clases destino a enfocar, separadas por comas, p. ej. "
                             f"Shooting,Assault,Road_Accident (opciones: {', '.join(TARGET_CLASSES)}). "
                             + what)


def parse_classes_or_exit(parser: argparse.ArgumentParser, raw):
    """parse_classes(), reportando una clase desconocida como error limpio de CLI."""
    try:
        return parse_classes(raw)
    except ValueError as exc:
        parser.error(str(exc))


def resolve_cookies(args, log, platforms=PLATFORMS) -> dict[str, str | None]:
    """{plataforma: ruta-o-None} a partir de las opciones --<plataforma>-cookies.
    Una ruta que no existe se avisa por log y se trata como 'sin cookies'."""
    cookies: dict[str, str | None] = {}
    for platform in platforms:
        path = getattr(args, f"{platform}_cookies", None)
        if path is None:
            cookies[platform] = None
        elif path.exists():
            cookies[platform] = str(path)
        else:
            log.warning("cookies file for %s not found: %s", platform, path)
            cookies[platform] = None
    return cookies
