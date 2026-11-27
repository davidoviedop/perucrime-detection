"""
lib/mil.py — puntuación de video en dos etapas, inspirada en Multiple-Instance
Learning (MIL), en lugar de un voto por mayoría simple.

Por qué no voto por mayoría: si el incidente real de un video solo ocupa una fracción
de un clip largo, un modelo que puntúa correctamente como Normal la mayoría tranquila
de las ventanas queda DERROTADO por el voto y el video se marca Normal: el modelo vio
el incidente y aun así se lo puntúa mal, justamente por acertar en las ventanas
aburridas. Es lo contrario de lo que se quiere de una función de puntuación.

Por qué tampoco max-pooling simple (el arreglo MIL clásico, p. ej. Sultani et al. en el
paper de UCF-Crime): esa formulación es esencialmente BINARIA (anómalo o no, un escalar
por segmento). Llevar "toma el máximo" directo a un softmax de 5 clases significa tomar
el máximo POR CLASE y discutir cuál es mayor, lo cual una sola ventana espuria puede
secuestrar, y ese riesgo crece con el largo del clip (más ventanas = más oportunidades
de que una falle). Ver el docstring de mil_aggregate_by_video: una detección binaria
estilo MIL (top-k promedio, ni máximo crudo ni voto) decide SI el video es anómalo, y un
voto por mayoría restringido a las ventanas que motivaron esa decisión decide CUÁL
crimen es, de modo que las ventanas Normal no diluyen el voto del tipo de crimen y una
ventana ruidosa no puede secuestrarlo sola.

Solo cambia cómo se agrupan las predicciones de ventana en un puntaje de video; el
entrenamiento sigue siendo a nivel de ventana.
"""
import math
from collections import Counter

import numpy as np


def mil_aggregate_by_video(records, probs, preds, labels, n_classes, normal_idx,
                            top_k_fraction=0.1, anomaly_threshold=0.5):
    """
    Puntuación de video en dos etapas, inspirada en MIL.

    records debe tener el MISMO largo y orden que probs/preds/labels: records[i] es el
    (vpath, start_frame, label) que produjo probs[i]/preds[i]/labels[i]. Es válido para
    los loaders de val/test (shuffle=False, así que la ventana i del dataset ES
    records[i] Y la predicción i), NO para el loader de entrenamiento (shuffle=True con
    reemplazo), que no tiene esa correspondencia.

    Etapa 1 — BINARIA: ¿el video es anómalo? Se agrupa el P(no Normal) de cada ventana
    con TOP-K PROMEDIO: se promedia el P(no Normal) de las k ventanas más anómalas, con
    k = max(1, ceil(top_k_fraction * n_ventanas)), que escala con la cantidad de
    ventanas del video. El top-k promedio (y no un máximo único) es el arreglo estándar
    para que un máximo crudo no lo secuestre una sola ventana ruidosa: necesita que
    VARIAS ventanas confiadas coincidan, no un pico espurio, pero aun así solo necesita
    una MINORÍA de las ventanas de un clip largo (a diferencia de un voto por mayoría,
    que necesita >50 %).

    Etapa 2 — MULTICLASE, condicionada a la etapa 1: si es anómalo, ¿cuál crimen? Voto
    por mayoría (los empates se resuelven por probabilidad media) entre las MISMAS k
    ventanas top de la etapa 1: las ventanas que hicieron ver anómalo al video son las
    que se consideran confiables para decir qué incidente es. Normal nunca compite en
    este voto, así que no puede derrotar a un incidente real pero breve como lo haría un
    voto por mayoría simple sobre TODAS las ventanas.

    Devuelve (video_preds, video_probs, video_anomaly_scores, video_labels,
    video_paths): una entrada por video distinto, en el orden en que aparecen.
      video_preds          — clase dura por video, de la decisión en dos etapas.
      video_probs          — vector de probabilidad por clase de cada video: el
                              promedio de los vectores de TODAS sus ventanas, un
                              resumen continuo para el AUC multiclase, independiente
                              de la decisión discreta de arriba. Lista de escalares
                              P(clase=1) si n_classes==2.
      video_anomaly_scores — el P(no Normal) top-k promedio por video, es decir, lo
                              que la etapa 1 umbraliza: pásalo a
                              binary_auc_from_scores() para un AUC Normal-vs-resto que
                              refleje la decisión real y no uno deducido de video_probs.
      video_labels, video_paths — etiqueta y ruta de cada video.
    """
    by_video: dict = {}
    order: list = []
    for (vpath, _start, _rec_label), p, pred, y in zip(records, probs, preds, labels):
        if n_classes == 2 and np.isscalar(p):
            p = [1.0 - p, p]
        if vpath not in by_video:
            order.append(vpath)
            by_video[vpath] = {"preds": [], "probs": [], "label": y}
        by_video[vpath]["preds"].append(pred)
        by_video[vpath]["probs"].append(p)

    video_preds, video_probs, video_anomaly_scores, video_labels = [], [], [], []
    for vpath in order:
        entry = by_video[vpath]
        win_probs = np.array(entry["probs"])         # (n_ventanas, n_clases)
        win_preds = entry["preds"]
        n_windows = len(win_preds)

        anomaly_scores = 1.0 - win_probs[:, normal_idx]         # P(no Normal) por ventana
        k = max(1, math.ceil(top_k_fraction * n_windows))
        top_idx = np.argsort(anomaly_scores)[-k:]               # las k ventanas más anómalas
        video_anomaly_score = float(anomaly_scores[top_idx].mean())

        if video_anomaly_score >= anomaly_threshold:
            candidate_preds = [win_preds[i] for i in top_idx if win_preds[i] != normal_idx]
            if not candidate_preds:
                # Cada una de las k ventanas top sigue siendo, individualmente,
                # Normal por argmax pese a tener el mayor P(no Normal) del video:
                # un caso ambiguo y de baja confianza. Se recurre a la predicción
                # de la ventana más anómala en lugar de forzar un voto sobre un
                # conjunto vacío.
                winner = win_preds[top_idx[-1]]
            else:
                vote_counts = Counter(candidate_preds)
                top_count   = max(vote_counts.values())
                tied        = [c for c, n in vote_counts.items() if n == top_count]
                mean_top_probs = win_probs[top_idx].mean(axis=0)
                winner = tied[0] if len(tied) == 1 else max(tied, key=lambda c: mean_top_probs[c])
        else:
            winner = normal_idx

        video_preds.append(int(winner))
        video_probs.append(win_probs.mean(axis=0).tolist())
        video_anomaly_scores.append(video_anomaly_score)
        video_labels.append(entry["label"])

    if n_classes == 2:
        video_probs = [p[1] for p in video_probs]   # de vuelta a escalar P(clase=1)

    return video_preds, video_probs, video_anomaly_scores, video_labels, order


# ── Piezas separadas de la puntuación (usadas por la ablación de la etapa 5) ────────────
#
# mil_aggregate_by_video() hace todo junto. Estas funciones separan sus tres operaciones para
# poder cambiar cada una por su cuenta; hard_decision() reproduce exactamente su decisión dura.
#
#   anomaly_score()  ¿qué tan anómalo es el video?        (AUC binario, umbral)
#   class_probs()    vector de probabilidades del video    (AUC multiclase)
#   hard_decision()  la regla actual en dos etapas         (F1, exactitud)
#   argmax_decision() alternativa: umbral + argmax de class_probs()

def top_k_indices(anomaly_scores, top_k_fraction):
    """Índices de las k ventanas más anómalas, con k = max(1, ceil(fracción × n)).
    fracción = 1.0 usa todas las ventanas; una fracción casi 0 (p. ej. 1e-9) usa solo la máxima."""
    k = max(1, math.ceil(top_k_fraction * len(anomaly_scores)))
    return np.argsort(anomaly_scores)[-k:]


def anomaly_score(win_probs, normal_idx, top_k_fraction):
    """Promedio de 1 − P(Normal) de las k ventanas más anómalas del video."""
    scores = 1.0 - np.asarray(win_probs)[:, normal_idx]
    return float(scores[top_k_indices(scores, top_k_fraction)].mean())


def class_probs(win_probs, normal_idx, top_k_fraction=1.0):
    """Probabilidades por clase del video: promedio de los vectores de las k ventanas más
    anómalas (fracción 1.0 = todas las ventanas, la forma actual del AUC multiclase)."""
    win_probs = np.asarray(win_probs)
    scores = 1.0 - win_probs[:, normal_idx]
    return win_probs[top_k_indices(scores, top_k_fraction)].mean(axis=0)


def hard_decision(win_probs, win_preds, normal_idx, top_k_fraction=0.1, anomaly_threshold=0.5):
    """La decisión dura de mil_aggregate_by_video: anómalo si el promedio top-k supera el umbral;
    la clase sale de un voto por mayoría entre esas k ventanas, sin que Normal vote."""
    win_probs = np.asarray(win_probs)
    scores = 1.0 - win_probs[:, normal_idx]
    top_idx = top_k_indices(scores, top_k_fraction)
    if float(scores[top_idx].mean()) < anomaly_threshold:
        return int(normal_idx)
    candidates = [win_preds[i] for i in top_idx if win_preds[i] != normal_idx]
    if not candidates:
        return int(win_preds[top_idx[-1]])
    votes = Counter(candidates)
    top = max(votes.values())
    tied = [c for c, n in votes.items() if n == top]
    mean_top = win_probs[top_idx].mean(axis=0)
    return int(tied[0] if len(tied) == 1 else max(tied, key=lambda c: mean_top[c]))


def argmax_decision(class_prob_vector, score, normal_idx, anomaly_threshold=0.5):
    """Normal si el video no supera el umbral; si lo supera, la clase no Normal de mayor probabilidad."""
    if score < anomaly_threshold:
        return int(normal_idx)
    p = np.array(class_prob_vector, dtype=float)
    p[normal_idx] = -1.0
    return int(np.argmax(p))
