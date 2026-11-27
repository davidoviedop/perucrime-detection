"""
ablation/scoring.py — matrices de puntuación por video M1–M12 (sin reentrenar).

Se aplican a las predicciones por ventana guardadas de las semillas de la referencia. La puntuación
binaria (¿es anómalo?) y la multiclase (¿qué delito?) son operaciones separadas (lib/mil.py) y se
ablacionan por separado; ver ablation_definition.md:

  M1–M5   agregación de anomalía: promedio de todas / máximo / top-k 5 % / 10 % (actual) / 20 %
          → cambian el AUC binario por video y, con la decisión dura, F1 y exactitud
  M6–M8   umbral 0.4 / 0.5 (actual) / 0.6, con top-k 10 %    → solo F1 y exactitud
  M9–M11  agregación multiclase: promedio de las probabilidades de todas / 10 % / 20 % más anómalas
          → cambian el AUC-Macro por video; decisión: Normal si el video no supera el umbral, si no
            el argmax de las clases no Normal
  M12     la decisión dura actual (umbral + voto entre las k ventanas, sin que Normal vote)
"""
import numpy as np

from lib.metrics import binary_auc_from_scores, classification_metrics, safe_auc
from lib.mil import anomaly_score, argmax_decision, class_probs, hard_decision

K_MAX = 1e-9   # fracción que deja k = 1 ventana (el máximo)

ANOMALY = {  # id: (descripción, fracción top-k)
    'M1': ('Promedio de todas las ventanas', 1.0),
    'M2': ('Máximo (1 ventana)', K_MAX),
    'M3': ('Promedio top-k 5 %', 0.05),
    'M4': ('Promedio top-k 10 % (actual)', 0.10),
    'M5': ('Promedio top-k 20 %', 0.20),
}
THRESHOLD = {'M6': 0.40, 'M7': 0.50, 'M8': 0.60}   # umbral, con top-k 10 %
MULTICLASS = {  # id: (descripción, fracción)
    'M9': ('Promedio de todas las ventanas (actual)', 1.0),
    'M10': ('Promedio del 10 % más anómalo', 0.10),
    'M11': ('Promedio del 20 % más anómalo', 0.20),
}
BASE_K, BASE_THR = 0.10, 0.50


def group_by_video(probs, preds, labels, recs):
    """[(win_probs, win_preds, label)] por video, en el orden en que aparecen."""
    order, by = [], {}
    for (vpath, _s, _l), p, pred, y in zip(recs, probs, preds, labels):
        if vpath not in by:
            order.append(vpath)
            by[vpath] = ([], [], y)
        by[vpath][0].append(p)
        by[vpath][1].append(int(pred))
    return [(np.array(by[v][0]), by[v][1], int(by[v][2])) for v in order]


def _hard_metrics(labels, preds, n_classes):
    m = classification_metrics(labels, preds, n_classes)
    return {'f1_macro': m['f1_macro'], 'accuracy': float(np.mean(np.array(labels) == np.array(preds)))}


def evaluate_all(videos, n_classes, normal_idx):
    """Métricas por video de M1–M12 para un conjunto de videos agrupados. Devuelve {id: {métrica: valor}}."""
    labels = [v[2] for v in videos]
    out = {}
    for mid, (_, frac) in ANOMALY.items():
        scores = [anomaly_score(wp, normal_idx, frac) for wp, _, _ in videos]
        preds = [hard_decision(wp, pr, normal_idx, frac, BASE_THR) for wp, pr, _ in videos]
        out[mid] = {'auc_binary': binary_auc_from_scores(labels, scores, normal_idx),
                    **_hard_metrics(labels, preds, n_classes)}
    scores_base = [anomaly_score(wp, normal_idx, BASE_K) for wp, _, _ in videos]
    for mid, thr in THRESHOLD.items():
        preds = [hard_decision(wp, pr, normal_idx, BASE_K, thr) for wp, pr, _ in videos]
        out[mid] = {'auc_binary': binary_auc_from_scores(labels, scores_base, normal_idx),
                    **_hard_metrics(labels, preds, n_classes)}
    for mid, (_, frac) in MULTICLASS.items():
        probs = np.array([class_probs(wp, normal_idx, frac) for wp, _, _ in videos])
        auc_macro, auc_bin = safe_auc(labels, probs, n_classes, normal_idx)
        preds = [argmax_decision(probs[i], scores_base[i], normal_idx, BASE_THR) for i in range(len(videos))]
        out[mid] = {'auc_macro': auc_macro, 'auc_binary_probs': auc_bin, **_hard_metrics(labels, preds, n_classes)}
    preds = [hard_decision(wp, pr, normal_idx, BASE_K, BASE_THR) for wp, pr, _ in videos]
    out['M12'] = _hard_metrics(labels, preds, n_classes)
    return out


def per_video_arrays(videos, normal_idx):
    """Puntajes por video (de las probabilidades de ventana, p. ej. promediadas entre semillas) para el bootstrap:
    anomalía de M1–M5 y vectores de clase de M9–M11."""
    labels = np.array([v[2] for v in videos])
    anomaly = {mid: np.array([anomaly_score(wp, normal_idx, frac) for wp, _, _ in videos])
               for mid, (_, frac) in ANOMALY.items()}
    probs = {mid: np.array([class_probs(wp, normal_idx, frac) for wp, _, _ in videos])
             for mid, (_, frac) in MULTICLASS.items()}
    return labels, anomaly, probs
