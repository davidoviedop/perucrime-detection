"""
lib/common/logger.py — configuración de logging compartida por los scripts de scraping.

Cada script crea dos salidas:
  1. stdout (nivel INFO): líneas de progreso limpias para la terminal.
  2. Un archivo con marca de tiempo en data/logs/ (nivel DEBUG): todo el
     detalle, incluidas las decisiones y su justificación. Es el rastro de
     auditoría del dataset.

Nombre del archivo: {nombre}_{YYYYMMDD_HHMMSS}.log

Uso
---
    from lib.common.logger import get_logger
    log, ruta_log = get_logger("pre_filter")
    log.info("Inicio")         # → stdout + archivo
    log.debug("Detalle: ...")  # → solo archivo
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(agent_name: str, logs_dir: str = "./data/logs"):
    """
    Crea y devuelve un logger configurado junto con la ruta de su archivo de log.

    Parámetros
    ----------
    agent_name : str
        Nombre del logger y prefijo del archivo de log.
    logs_dir : str
        Carpeta de los logs. Se crea si no existe.

    Devuelve
    --------
    (logging.Logger, Path)
        El logger y la ruta de su archivo de log.
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_path / f"{agent_name}_{timestamp}.log"

    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.DEBUG)

    # ── Handler de archivo: todo el detalle (DEBUG) ────────────────────
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # ── Handler de stdout: solo progreso (INFO) ────────────────────────
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger, log_file
