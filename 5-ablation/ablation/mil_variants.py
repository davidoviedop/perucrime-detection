"""
ablation/mil_variants.py — formas alternativas de puntuar un VIDEO a partir de las
predicciones de sus ventanas (ablación de la puntuación MIL de lib/mil.py).

No hace falta reentrenar: se calculan a partir de las probabilidades por ventana que guarda
cada corrida (preds_{split}.npz). Todas devuelven lo mismo que mil_aggregate_by_video:
(preds, probs, anomaly_scores, labels, paths), con `probs` = promedio de las probabilidades
de las ventanas del video (para el AUC multiclase) y `anomaly_scores` = el puntaje continuo
con el que se decide si el video es anómalo (para el AUC binario).

  mil          la referencia: top-k promedio de P(no Normal) + voto entre esas k ventanas
  vote_all     voto por mayoría de TODAS las ventanas (el esquema ingenuo que MIL reemplaza)
  mean_prob    clase con mayor probabilidad media del video
  max_pool     P(no Normal) máxima; la clase es la de mayor probabilidad máxima entre las no Normal
"""
from collections import Counter

import numpy as np

from lib.metrics import binary_auc_from_scores
from lib.mil import mil_aggregate_by_video
from lib.scoring import scoring_dict


def _group(records, probs, preds, labels):
    by_video, order = {}, []
    for (vpath, _s, _l), p, pred, y in zip(records, probs, preds, labels):
        if vpath not in by_video:
            order.append(vpath)
            by_video[vpath] = {'probs': [], 'preds': [], 'label': y}
        by_video[vpath]['probs'].append(p)
        by_video[vpath]['preds'].append(pred)
    return by_video, order


def aggregate(kind, records, probs, preds, labels, n_classes, normal_idx, top_k_fraction=0.1, threshold=0.5):
    if kind == 'mil':
        return mil_aggregate_by_video(records, probs.tolist(), preds.tolist(), labels.tolist(), n_classes,
                                      normal_idx, top_k_fraction=top_k_fraction, anomaly_threshold=threshold)
    by_video, order = _group(records, probs, preds, labels)
    v_preds, v_probs, v_scores, v_labels = [], [], [], []
    for vpath in order:
        e = by_video[vpath]
        wp = np.array(e['probs'])                    # (n_ventanas, n_clases)
        mean_p = wp.mean(axis=0)
        anomaly = 1.0 - wp[:, normal_idx]
        if kind == 'vote_all':
            votes = Counter(e['preds'])
            top = max(votes.values())
            tied = [c for c, n in votes.items() if n == top]
            winner = tied[0] if len(tied) == 1 else max(tied, key=lambda c: mean_p[c])
            score = float(anomaly.mean())
        elif kind == 'mean_prob':
            winner = int(np.argmax(mean_p))
            score = float(anomaly.mean())
        elif kind == 'max_pool':
            score = float(anomaly.max())
            per_class_max = wp.max(axis=0).copy()
            per_class_max[normal_idx] = -1.0
            winner = int(np.argmax(per_class_max)) if score >= threshold else normal_idx
        else:
            raise ValueError(f'puntuación de video desconocida: {kind!r}')
        v_preds.append(int(winner)); v_probs.append(mean_p.tolist())
        v_scores.append(score); v_labels.append(e['label'])
    return v_preds, v_probs, v_scores, v_labels, order


def video_metrics(kind, records, probs, preds, labels, classes, normal_idx, **params):
    """Métricas por video (las mismas de lib/evaluation.py) con la puntuación `kind`."""
    v_preds, v_probs, v_scores, v_labels, _ = aggregate(
        kind, records, probs, preds, labels, len(classes), normal_idx, **params)
    flat, _, _ = scoring_dict('video_', v_labels, v_preds, v_probs, 'video', classes, normal_idx,
                              binary_auc_override=binary_auc_from_scores(v_labels, v_scores, normal_idx))
    flat['video_accuracy'] = float(np.mean([p == l for p, l in zip(v_preds, v_labels)]))
    flat['video_count'] = len(v_labels)
    return flat
