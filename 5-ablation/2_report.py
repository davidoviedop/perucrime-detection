"""
2_report.py — PASO 2: tabla de resultados de la ablación (CSV + Markdown en results/).

Junta las corridas terminadas de outputs/{suite}/ y reporta, por variante, las métricas de
test (media ± desviación entre semillas) y su diferencia con la referencia.

Uso
---
    python 2_report.py
    python 2_report.py --suite config/ablation_mobilevit_xs.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 5-ablation/

from ablation import OUTPUTS_DIR, RESULTS_DIR, STAGE_DIR
from ablation.report import aggregate, collect, per_class_recall, write
from ablation.suite import load_suite


def main():
    parser = argparse.ArgumentParser(description="Tabla de resultados de la ablación")
    parser.add_argument("--suite", type=Path, default=STAGE_DIR / "config" / "ablation_mobilevit_xs.yaml")
    parser.add_argument("--outputs-dir", type=Path, default=None)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    outputs_dir = args.outputs_dir or OUTPUTS_DIR / suite["name"]
    rows = collect(outputs_dir)
    if not rows:
        raise SystemExit(f"no hay corridas terminadas en {outputs_dir} (corre primero 1_run_ablation.py)")
    order = [v["name"] for v in suite["variants"]]
    agg = aggregate(rows, order, suite["reference"])
    classes, recall = per_class_recall(rows, order)
    csv_path, md_path = write(suite["name"], agg, classes, recall, RESULTS_DIR, suite["reference"])
    print(md_path.read_text(encoding="utf-8"))
    print(f"Guardado: {csv_path}\n          {md_path}")
    missing = [n for n in order if n not in {a["variant"] for a in agg}]
    if missing:
        print(f"Sin resultados todavía: {missing}")


if __name__ == "__main__":
    main()
