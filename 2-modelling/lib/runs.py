"""
lib/runs.py — carpeta de una corrida: outputs/{run_id}/

    last.pt       checkpoint completo de la última época (para reanudar)
    best.pt       state_dict del mejor modelo según val_auc
    run.yaml      configuración + clases con las que se entrenó (lo lee 2_evaluate.py)
    history.csv   métricas por época
    metrics_*.json  métricas finales de evaluación
"""
import secrets
import string
from pathlib import Path

import yaml

from .config import Config

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def new_run_id() -> str:
    """Id corto de 8 caracteres (minúsculas y dígitos), como los ids de wandb."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


def run_dir(run_id: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return Path(outputs_dir) / run_id


def save_run_info(out_dir: Path, cfg: Config, classes: list[str], sources: list[str], run_id: str) -> None:
    info = {"run_id": run_id, "classes": classes, "sources": sources, "config": cfg.to_dict()}
    (out_dir / "run.yaml").write_text(yaml.safe_dump(info, sort_keys=False, allow_unicode=True))


def load_run_info(out_dir: Path) -> tuple[Config, list[str], list[str]]:
    """(configuración, clases, fuentes) de una corrida ya entrenada."""
    info = yaml.safe_load((Path(out_dir) / "run.yaml").read_text())
    return Config.from_dict(info["config"]), info["classes"], info["sources"]
