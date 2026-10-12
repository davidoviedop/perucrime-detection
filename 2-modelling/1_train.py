"""
1_train.py — PASO 1: entrena el detector de anomalías (backbone por fotograma +
transformer temporal) sobre las ventanas del dataset.

Una corrida = una carpeta outputs/{run_id}/ con
  best.pt, last.pt   mejor modelo (según val_auc por ventana) y último checkpoint
  run.yaml           configuración y clases con las que se entrenó
  history.csv        métricas por época (ventana y video MIL en validación)

El entrenamiento es a nivel de ventana. La puntuación MIL por video (lib/mil.py) solo
cambia cómo se reportan las métricas de validación, no el entrenamiento; ver README.md.

Si la carpeta de la corrida ya tiene best.pt se da por entrenada y no se entrena de nuevo
(usa --retrain para forzarlo); si solo tiene last.pt, se reanuda desde ahí.

Uso
---
    python 1_train.py                                   # profesor ConvNeXtV2-base (config/teacher_convnextv2_base.yaml)
    python 1_train.py --set epochs=40 --wandb           # entrenamiento largo con wandb
    python 1_train.py --config config/mi_config.yaml --run-id abc12345
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 2-modelling/

from lib.cli import STAGE_DIR, add_override_args, add_tracking_args, parse_overrides
from lib.config import Config
from lib.pipeline import train_run
from lib.runs import OUTPUTS_DIR, new_run_id


def main():
    parser = argparse.ArgumentParser(description="Entrena el detector de anomalías")
    parser.add_argument("--config", type=Path, default=STAGE_DIR / "config" / "teacher_convnextv2_base.yaml",
                        help="YAML con los hiperparámetros (por defecto: el profesor ConvNeXtV2-base)")
    parser.add_argument("--run-id", default=None,
                        help="Id de la corrida (por defecto se genera uno). Si ya existe outputs/{id}/ se reutiliza o reanuda")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR,
                        help="Carpeta donde se guardan las corridas (por defecto: outputs/)")
    parser.add_argument("--retrain", action="store_true",
                        help="Entrena aunque la corrida ya tenga best.pt")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="No descarga los pesos preentrenados del backbone (útil para pruebas)")
    add_override_args(parser)
    add_tracking_args(parser)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config, **parse_overrides(args))
    run_id = args.run_id or new_run_id()
    train_run(cfg, run_id, args.outputs_dir, use_wandb=args.wandb, retrain=args.retrain,
              pretrained=False if args.no_pretrained else None)
    print(f'\nSiguiente paso: python 2_evaluate.py --run-id {run_id}')


if __name__ == "__main__":
    main()
