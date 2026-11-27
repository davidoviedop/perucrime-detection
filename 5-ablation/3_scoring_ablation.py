"""
3_scoring_ablation.py — PASO 3 (sin entrenar): ablación de la puntuación por video, M1–M12.

Toma las predicciones por ventana (preds_{val,test}.npz) de las semillas de la referencia (A0) y
re-puntúa los videos con las matrices de ablation/scoring.py. No usa GPU ni repite la inferencia.

  M1–M5   agregación de anomalía     M6–M8  umbral     M9–M11  agregación multiclase     M12  decisión dura actual

Se reporta media ± desviación entre semillas (val y test); para el AUC binario de M1–M5 y el AUC-Macro de
M10–M11 también el Δ frente a la regla actual (M4 y M9) con IC 95 % por bootstrap pareado por video, sobre
las probabilidades promediadas entre semillas. El umbral (M6–M8) es un punto de operación: se elige en
VALIDACIÓN (mayor F1-Macro) y el test solo se reporta.

Uso
---
    python 3_scoring_ablation.py                       # A0 de outputs/modelling/
    python 3_scoring_ablation.py --run-dirs outputs/modelling/A0_s42 outputs/modelling/A0_s43 outputs/modelling/A0_s44
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 5-ablation/
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))   # raíz de workspace/ (run_report.py)
from run_report import RunReport, current

import numpy as np
from sklearn.metrics import roc_auc_score

from ablation import OUTPUTS_DIR, RESULTS_DIR
from ablation.report import fmt
from ablation.scoring import ANOMALY, MULTICLASS, THRESHOLD, evaluate_all, group_by_video, per_video_arrays
from ablation.stats import auc_macro, paired_delta_ci, stratified_video_draws
from lib.evaluation import load_predictions
from lib.runs import load_run_info
from ablation.report import archive


def seed_averaged_videos(run_dirs, split):
    runs = [load_predictions(d / f'preds_{split}.npz') for d in run_dirs]
    recs = runs[0][3]
    if any(r[3] != recs for r in runs[1:]):
        raise SystemExit('las semillas tienen ventanas distintas')
    probs = np.mean([r[0] for r in runs], axis=0)
    return group_by_video(probs, runs[0][1], runs[0][2], recs)


def main():
    parser = argparse.ArgumentParser(description="Ablación de la puntuación por video (M1–M12)")
    parser.add_argument("--run-dirs", nargs="+", type=Path, default=None,
                        help="Corridas de la referencia (por defecto: outputs/modelling/A0_s*)")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    run_dirs = [d.resolve() for d in (args.run_dirs or sorted((OUTPUTS_DIR / "modelling").glob("A0_s*")))]
    if not run_dirs:
        raise SystemExit("no hay corridas de A0 (corre primero: python 1_run_ablation.py --only A0)")
    cfg, classes, _ = load_run_info(run_dirs[0])
    normal_idx, n_classes = classes.index("Normal"), len(classes)
    print(f"Referencia: {[d.name for d in run_dirs]}  ({n_classes} clases)")

    results = {}   # (split, mid) -> {métrica: [valor por semilla]}
    for split in ("val", "test"):
        for d in run_dirs:
            probs, preds, labels, recs = load_predictions(d / f"preds_{split}.npz")
            res = evaluate_all(group_by_video(probs, preds, labels, recs), n_classes, normal_idx)
            for mid, mets in res.items():
                for k, v in mets.items():
                    results.setdefault((split, mid), {}).setdefault(k, []).append(v)

    def ms(split, mid, key):
        v = np.array(results[(split, mid)].get(key, [np.nan]), dtype=float)
        return fmt(float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"))

    # bootstrap pareado sobre las probabilidades promediadas entre semillas (test)
    videos = seed_averaged_videos(run_dirs, "test")
    labels, anomaly, mprobs = per_video_arrays(videos, normal_idx)
    draws = stratified_video_draws(labels, args.n_boot)

    def anom_auc(mid):
        return lambda ids: float("nan") if ids is not None and len(set((labels[ids] != normal_idx).tolist())) < 2 else \
            roc_auc_score((labels != normal_idx).astype(int)[ids if ids is not None else slice(None)],
                          anomaly[mid][ids if ids is not None else slice(None)])

    def macro_auc(mid):
        return lambda ids: auc_macro(labels if ids is None else labels[ids], mprobs[mid] if ids is None else mprobs[mid][ids])

    L = [f"# Ablación de la puntuación por video (M1–M12) — referencia `{run_dirs[0].name.rsplit('_s', 1)[0]}`", "",
         f"{len(run_dirs)} semillas; media ± desviación. Δ con IC 95 % por bootstrap pareado por video "
         f"({args.n_boot} remuestreos estratificados), sobre las probabilidades promediadas entre semillas.", ""]

    L += ["## Agregación de anomalía (M1–M5): cambia el AUC binario por video", "",
          "| ID | Agregación | AUC binario test | Δ vs M4 [IC 95 %] | F1-Macro test | Exactitud test | AUC binario val |", "|---|---|---|---|---|---|---|"]
    for mid, (desc, _) in ANOMALY.items():
        delta = "—" if mid == "M4" else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*paired_delta_ci(anom_auc("M4"), anom_auc(mid), draws))
        L.append(f"| {mid} | {desc} | {ms('test', mid, 'auc_binary')} | {delta} | {ms('test', mid, 'f1_macro')} | "
                 f"{ms('test', mid, 'accuracy')} | {ms('val', mid, 'auc_binary')} |")

    best = max(THRESHOLD, key=lambda m: np.nanmean(results[("val", m)]["f1_macro"]))
    L += ["", "## Umbral (M6–M8): no cambia ningún AUC; punto de operación", "",
          f"Elegido en validación (mayor F1-Macro): **{best}** (umbral {THRESHOLD[best]:.2f}).", "",
          "| ID | Umbral | F1-Macro val | Exactitud val | F1-Macro test | Exactitud test |", "|---|---|---|---|---|---|"]
    for mid, thr in THRESHOLD.items():
        L.append(f"| {mid}{' ✓' if mid == best else ''} | {thr:.2f}{' (actual)' if mid == 'M7' else ''} | {ms('val', mid, 'f1_macro')} | "
                 f"{ms('val', mid, 'accuracy')} | {ms('test', mid, 'f1_macro')} | {ms('test', mid, 'accuracy')} |")

    L += ["", "## Agregación multiclase (M9–M11): cambia el AUC-Macro por video", "",
          "| ID | Ventanas usadas | AUC-Macro test | Δ vs M9 [IC 95 %] | F1-Macro test | Exactitud test | AUC-Macro val |", "|---|---|---|---|---|---|---|"]
    for mid, (desc, _) in MULTICLASS.items():
        delta = "—" if mid == "M9" else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*paired_delta_ci(macro_auc("M9"), macro_auc(mid), draws))
        L.append(f"| {mid} | {desc} | {ms('test', mid, 'auc_macro')} | {delta} | {ms('test', mid, 'f1_macro')} | "
                 f"{ms('test', mid, 'accuracy')} | {ms('val', mid, 'auc_macro')} |")

    L += ["", "## Decisión dura de referencia (M12) frente al argmax de M9–M11", "",
          "| ID | Decisión | F1-Macro test | Exactitud test | F1-Macro val | Exactitud val |", "|---|---|---|---|---|---|",
          f"| M12 | Umbral + voto entre las k ventanas, sin que Normal vote (actual) | {ms('test', 'M12', 'f1_macro')} | "
          f"{ms('test', 'M12', 'accuracy')} | {ms('val', 'M12', 'f1_macro')} | {ms('val', 'M12', 'accuracy')} |"]
    for mid, (desc, _) in MULTICLASS.items():
        L.append(f"| {mid} | Normal si no supera el umbral; si no, argmax no Normal ({desc.split(' (')[0].lower()}) | "
                 f"{ms('test', mid, 'f1_macro')} | {ms('test', mid, 'accuracy')} | {ms('val', mid, 'f1_macro')} | {ms('val', mid, 'accuracy')} |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "scoring_M1-M12.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    copies = archive([path], RESULTS_DIR)
    print("\n".join(L))
    current().set_config("scoring", {"run_dirs": [str(d) for d in run_dirs], "n_boot": args.n_boot})
    for d in run_dirs:
        current().add_input(d.name, d)
    current().add_results({f"{split}.{mid}.{k}": float(np.nanmean(v)) for (split, mid), mets in results.items()
                           for k, v in mets.items()}, section="metricas_medias_entre_semillas")
    current().add_output(path)
    current().add_output(copies[0], "copia con fecha")
    print(f"\nGuardado en {path}")


if __name__ == "__main__":
    with RunReport(Path(__file__).resolve().parent, "3_scoring_ablation"):
        main()
