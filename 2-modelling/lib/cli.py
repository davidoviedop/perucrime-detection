"""lib/cli.py — opciones de línea de comandos comunes a los scripts de la etapa."""
import argparse
from pathlib import Path

import yaml

STAGE_DIR = Path(__file__).resolve().parents[1]


def add_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--set", action="append", default=[], metavar="CAMPO=VALOR",
                        help="Sobrescribe un campo de la configuración, p. ej. --set epochs=40 "
                             "--set batch_size=8 (se puede repetir)")
    parser.add_argument("--data-root", default=None,
                        help="Carpeta del dataset, {fuente}/{split}/{clase}/ (pisa data_root de la configuración)")


def parse_overrides(args) -> dict:
    """{campo: valor} a partir de --set y --data-root; los valores se interpretan como YAML."""
    out = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set espera CAMPO=VALOR, recibió: {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    if args.data_root is not None:
        out["data_root"] = args.data_root
    return out


def add_tracking_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb", action="store_true",
                        help="También registra la corrida en Weights & Biases (requiere wandb)")


def resolve_data_root(data_root: str) -> Path:
    """Una ruta relativa se interpreta desde la carpeta de la etapa, no desde donde se ejecute."""
    p = Path(data_root)
    return p if p.is_absolute() else (STAGE_DIR / p).resolve()
