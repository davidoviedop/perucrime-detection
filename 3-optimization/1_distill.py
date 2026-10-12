"""
1_distill.py — PASO 1: destila el profesor (etapa 2) en un estudiante pequeño.

El estudiante (MobileViT-XS + el mismo transformer temporal) se entrena con una mezcla de
la pérdida con las etiquetas y la divergencia KL contra las salidas suavizadas del
profesor (ver optimization/distill.py). El profesor es el modelo completo, sin poda ni
cuantización; esas compresiones se aplican después, solo al estudiante (2_compress.py).

El entrenamiento, los checkpoints, la reanudación y la validación con puntuación MIL son
los mismos de la etapa 2 (lib/pipeline.py); aquí solo cambian el modelo, la pérdida y el
profesor.

Una corrida = outputs/{run_id}/ con best.pt, last.pt, history.csv y run.yaml (la sección
`extra` guarda los parámetros de destilación). Esa carpeta se puede evaluar con
`python ../2-modelling/2_evaluate.py --run-id <id> --outputs-dir outputs`.

Uso
---
    python 1_distill.py                                        # estudiante de referencia
    python 1_distill.py --teacher-run-dir ../2-modelling/outputs/<id> --set epochs=40 --wandb
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 3-optimization/ (agrega también 2-modelling/)

from optimization import STAGE_DIR   # al importar `optimization` queda lib/ de la etapa 2 en sys.path
from lib.cli import add_override_args, add_tracking_args, parse_overrides
from lib.pipeline import train_run
from lib.runs import load_run_info, new_run_id
from optimization.config import DistillConfig
from optimization.distill import DistillationLoss, KDTrainWrapper, load_teacher


def main():
    parser = argparse.ArgumentParser(description="Destila el profesor en un estudiante")
    parser.add_argument("--config", type=Path, default=STAGE_DIR / "config" / "student_mobilevit_xs_distill.yaml",
                        help="YAML con los hiperparámetros (por defecto: estudiante MobileViT-XS de referencia)")
    parser.add_argument("--teacher-run-dir", default=None,
                        help="Carpeta de la corrida del profesor (pisa teacher_run_dir de la configuración)")
    parser.add_argument("--run-id", default=None, help="Id de la corrida (por defecto se genera uno)")
    parser.add_argument("--outputs-dir", type=Path, default=STAGE_DIR / "outputs",
                        help="Carpeta donde se guardan las corridas (por defecto: outputs/)")
    parser.add_argument("--retrain", action="store_true", help="Entrena aunque la corrida ya tenga best.pt")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="No descarga los pesos preentrenados del backbone (útil para pruebas)")
    add_override_args(parser)
    add_tracking_args(parser)
    args = parser.parse_args()

    overrides = parse_overrides(args)
    if args.teacher_run_dir is not None:
        overrides["teacher_run_dir"] = args.teacher_run_dir
    dcfg = DistillConfig.from_yaml(args.config, **overrides)
    if not dcfg.teacher_run_dir:
        raise SystemExit("falta el profesor: usa --teacher-run-dir o teacher_run_dir en la configuración")
    teacher_dir = Path(dcfg.teacher_run_dir)
    if not teacher_dir.is_absolute():
        teacher_dir = (STAGE_DIR / teacher_dir).resolve()
    if not (teacher_dir / "best.pt").exists():
        raise SystemExit(f"el profesor no tiene {teacher_dir / 'best.pt'}: entrénalo con 2-modelling/1_train.py")

    cfg, distill_params = dcfg.split()
    criterion = DistillationLoss(dcfg.kd_temperature, dcfg.kd_alpha, label_smoothing=cfg.label_smoothing)
    print(f'Loss: alpha*CE + (1-alpha)*T^2*KL  alpha={dcfg.kd_alpha}  T={dcfg.kd_temperature}  '
          f'(CE label_smoothing={cfg.label_smoothing}; profesor {teacher_dir.name} -> estudiante {cfg.backbone})')

    def wrap_train_model(student, device):
        teacher, t_cfg, t_classes = load_teacher(teacher_dir, device)
        n_student = student.head[-1].out_features
        if len(t_classes) != n_student:
            raise SystemExit(f"el profesor tiene {len(t_classes)} clases y el estudiante {n_student}")
        n_teacher = sum(p.numel() for p in teacher.parameters())
        n_student_p = sum(p.numel() for p in student.parameters())
        print(f'Teacher loaded from {teacher_dir}: {n_teacher:,} params  ->  student {n_student_p:,} params')
        train_model = KDTrainWrapper(student, teacher)
        criterion.source = train_model
        return train_model

    run_id = args.run_id or new_run_id()
    teacher_cfg, _, _ = load_run_info(teacher_dir)
    extra = {"distillation": {**distill_params, "teacher_run_dir": str(teacher_dir),
                              "teacher_backbone": teacher_cfg.backbone}}
    train_run(cfg, run_id, args.outputs_dir, use_wandb=args.wandb, retrain=args.retrain,
              pretrained=False if args.no_pretrained else None, criterion=criterion,
              wrap_train_model=wrap_train_model, extra_info=extra,
              extra_config={**distill_params, "distillation": True, "teacher_backbone": teacher_cfg.backbone})
    print(f'\nSiguiente paso: python 2_compress.py --run-dir {args.outputs_dir}/{run_id}')


if __name__ == "__main__":
    main()
