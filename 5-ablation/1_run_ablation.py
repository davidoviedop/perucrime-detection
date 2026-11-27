"""
1_run_ablation.py — PASO 1: entrena y evalúa cada variante de una suite de ablación.

Cada (variante, semilla) es una corrida de la etapa 2 en outputs/{suite}/{id}_s{semilla}/ (ver
ablation/suite.py). Se puede interrumpir y volver a lanzar: las terminadas se omiten y las
interrumpidas se reanudan.

Uso
---
    python 1_run_ablation.py --suite config/pilot_lr.yaml        # piloto de LR (solo validación)
    python 1_run_ablation.py --list                              # qué se va a correr; no corre nada
    python 1_run_ablation.py --only A0                           # A0 con sus 3 semillas
    python 1_run_ablation.py --only A10 A13 --seeds 42           # solo esas variantes (A0 se incluye siempre)
    python 1_run_ablation.py --set epochs=2                      # prueba rápida del flujo
    python 1_run_ablation.py                                     # la suite completa (42 corridas)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 5-ablation/ (agrega también 2-modelling/)
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))   # raíz de workspace/ (run_report.py)
from run_report import RunReport, current

from ablation import OUTPUTS_DIR, STAGE_DIR
from lib.cli import add_override_args, add_tracking_args, parse_overrides
from ablation.suite import expand, load_suite, make_config, run_suite


def main():
    parser = argparse.ArgumentParser(description="Corre una suite de ablación")
    parser.add_argument("--suite", type=Path, default=STAGE_DIR / "config" / "ablation_modelling.yaml",
                        help="YAML de la suite (por defecto: config/ablation_modelling.yaml)")
    parser.add_argument("--only", nargs="+", default=None, metavar="ID",
                        help="Solo estas variantes (la referencia se incluye siempre)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Semillas (pisa las de la suite)")
    parser.add_argument("--outputs-dir", type=Path, default=None,
                        help="Carpeta de salida (por defecto: outputs/{nombre de la suite}/)")
    parser.add_argument("--retrain", action="store_true", help="Repite las corridas aunque ya estén terminadas")
    parser.add_argument("--list", action="store_true", help="Lista las corridas y sus cambios, y sale")
    add_override_args(parser)
    add_tracking_args(parser)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    specs = expand(suite, args.only, args.seeds)
    cli_overrides = parse_overrides(args)
    outputs_dir = args.outputs_dir or OUTPUTS_DIR / suite["name"]

    print(f"Suite {suite['name']}: {len(specs)} corridas  →  {outputs_dir}")
    ref_spec = next(s for s in specs if s.vid == suite["reference"])
    ref_cfg = make_config(suite, ref_spec, cli_overrides).to_dict()
    for s in specs:
        cfg = make_config(suite, s, cli_overrides).to_dict()      # también valida los campos del YAML
        diff = {k: v for k, v in cfg.items() if ref_cfg.get(k) != v and k not in ("seed", "wandb_run_name")}
        print(f"  {s.run_id:<10} cambia: {str(diff or '—'):<48} {s.question}")
    if args.list:
        return

    failed = run_suite(suite, specs, outputs_dir, cli_overrides, use_wandb=args.wandb, retrain=args.retrain)
    if failed:
        print(f"\nFallaron {len(failed)} corridas: {failed}")
        sys.exit(1)
    print(f"\nListo. Siguiente paso: python 2_report.py --suite {args.suite}")


if __name__ == "__main__":
    with RunReport(Path(__file__).resolve().parent, "1_run_ablation"):
        main()
