"""
2_report.py — PASO 2: tablas de la ablación (CSV + Markdown en results/).

Junta las corridas terminadas de outputs/{suite}/: media ± desviación entre semillas, Δ frente a la
referencia con IC 95 % por bootstrap pareado por video, veredicto y métricas de validación.
Para el piloto de LR (suite con solo validación) imprime la comparación de validación.

Uso
---
    python 2_report.py
    python 2_report.py --suite config/pilot_lr.yaml
    python 2_report.py --n-boot 500 --no-window-ci       # más rápido
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 5-ablation/
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))   # raíz de workspace/ (run_report.py)
from run_report import RunReport, current

from ablation import OUTPUTS_DIR, RESULTS_DIR, STAGE_DIR
from ablation.report import analyze, class_counts, collect, fmt, mean_sd, write
from run_report import current
from ablation.suite import load_suite


def pilot_report(runs, order):
    print("| ID | Pregunta | AUC-Macro ventana (val) | AUC-Macro video (val) | Mejor época |\n|---|---|---|---|---|")
    for vid in order:
        for r in runs.get(vid, []):
            m = r["val"]
            print(f"| {vid} | {r['info'].get('question', '')} | {m['val_window_auc_macro']:.4f} | "
                  f"{m['val_video_auc_macro']:.4f} | {r['info'].get('best_epoch', '—')} |")
    print("\nSi el LR 3e-5 queda por debajo de 1e-5 en validación, toda la suite usa 1e-5 (lr_backbone en a0_convnextv2_femto.yaml).")


def main():
    parser = argparse.ArgumentParser(description="Tablas de resultados de la ablación")
    parser.add_argument("--suite", type=Path, default=STAGE_DIR / "config" / "ablation_modelling.yaml")
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=2000, help="Remuestreos del bootstrap (por defecto 2000)")
    parser.add_argument("--no-window-ci", action="store_true", help="No calcula el IC del AUC por ventana (más rápido)")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    outputs_dir = args.outputs_dir or OUTPUTS_DIR / suite["name"]
    runs = collect(outputs_dir)
    if not runs:
        raise SystemExit(f"no hay corridas terminadas en {outputs_dir} (corre primero 1_run_ablation.py)")
    order = [v["id"] for v in suite["variants"]]

    if "test" not in suite["eval_splits"]:
        pilot_report(runs, order)
        return
    ref = suite["reference"]
    if ref not in runs:
        raise SystemExit(f"falta la referencia {ref} en {outputs_dir}")
    rows, classes = analyze(runs, order, ref, n_boot=args.n_boot, window_ci=not args.no_window_ci)
    path = write(suite["name"], rows, classes, class_counts(runs[ref]), RESULTS_DIR, ref, args.n_boot)
    print(path.read_text(encoding="utf-8"))
    print(f"Guardado: {path}")
    current().set_config("suite", {"suite": suite["name"], "reference": ref, "n_boot": args.n_boot, "window_ci": not args.no_window_ci})
    current().add_input("outputs_dir", outputs_dir)
    current().add_results({r["id"]: r["verdict"] for r in rows}, section="veredictos")
    current().add_results({r["id"]: r["delta_video_auc"] for r in rows if "delta_video_auc" in r}, section="delta_auc_macro_video_vs_referencia")
    current().add_results({r["id"]: r["test_video_auc_macro"] for r in rows}, section="auc_macro_video_test_media")
    current().add_output(path)
    current().add_output(path.with_suffix(".csv"))
    current().add_output(RESULTS_DIR / "history", "copias con fecha de estas tablas")
    missing = [v for v in order if v not in runs]
    if missing:
        print(f"Sin resultados todavía: {missing}")
    short = [(v, len(r)) for v, r in runs.items() if len(r) < len(suite['seeds'])]
    if short:
        print(f"Con menos semillas de las previstas: {short}")


if __name__ == "__main__":
    with RunReport(Path(__file__).resolve().parent, "2_report"):
        main()
