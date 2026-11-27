"""
ablation/stats.py — análisis estadístico de la ablación: probabilidades por video promediadas
entre semillas y bootstrap pareado por video.

El test es pequeño (58 videos), así que cada Δ se acompaña de un intervalo de confianza:
se remuestrean VIDEOS (nunca ventanas sueltas: las ventanas de un video no son independientes),
estratificados por clase para que ninguna clase desaparezca en un remuestreo, y los MISMOS
videos remuestreados se usan para la referencia y para la variante (bootstrap pareado).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from lib.evaluation import load_predictions


def auc_macro(labels, probs):
    """AUC uno-contra-resto promediado sobre las clases presentes (nan si ninguna es evaluable)."""
    labels = np.asarray(labels)
    out = []
    for c in range(probs.shape[1]):
        y = (labels == c).astype(int)
        if 0 < y.sum() < len(y):
            out.append(roc_auc_score(y, probs[:, c]))
    return float(np.mean(out)) if out else float('nan')


class SeedAveragedPredictions:
    """Predicciones de una configuración: probabilidades por ventana promediadas entre semillas.

    Todas las semillas deben tener las mismas ventanas (mismo dataset y misma definición de
    ventana); si no, falla. `video_probs` es el promedio de las ventanas de cada video (M9).
    """

    def __init__(self, npz_paths, videos=None):
        """videos: si se indica, solo se usan las ventanas de esos videos (para comparar sobre un conjunto común)."""
        runs = [load_predictions(p) for p in npz_paths]
        _, _, labels, recs = runs[0]
        for _, _, _, r in runs[1:]:
            if r != recs:
                raise ValueError('las semillas tienen ventanas distintas; no se pueden promediar')
        self.n_seeds = len(runs)
        self.window_probs = np.mean([r[0] for r in runs], axis=0)
        self.window_labels = np.asarray(labels)
        self.window_video = np.array([r[0] for r in recs])
        if videos is not None:
            keep = np.isin(self.window_video, list(videos))
            self.window_probs, self.window_labels, self.window_video = (
                self.window_probs[keep], self.window_labels[keep], self.window_video[keep])
        self.videos = sorted(set(self.window_video.tolist()))
        index = {v: i for i, v in enumerate(self.videos)}
        self.window_video_idx = np.array([index[v] for v in self.window_video])
        self.video_labels = np.zeros(len(self.videos), dtype=int)
        self.video_probs = np.zeros((len(self.videos), self.window_probs.shape[1]))
        self.windows_of_video = []
        for i in range(len(self.videos)):
            w = np.where(self.window_video_idx == i)[0]
            self.windows_of_video.append(w)
            self.video_labels[i] = self.window_labels[w[0]]
            self.video_probs[i] = self.window_probs[w].mean(axis=0)

    def video_auc(self, video_ids=None):
        ids = np.arange(len(self.videos)) if video_ids is None else video_ids
        return auc_macro(self.video_labels[ids], self.video_probs[ids])

    def window_auc(self, video_ids=None):
        ids = np.arange(len(self.videos)) if video_ids is None else video_ids
        w = np.concatenate([self.windows_of_video[i] for i in ids])
        return auc_macro(self.window_labels[w], self.window_probs[w])


def stratified_video_draws(video_labels, n_boot, seed=0):
    """n_boot remuestreos de índices de video, con reemplazo dentro de cada clase."""
    rng = np.random.default_rng(seed)
    by_class = [np.where(video_labels == c)[0] for c in np.unique(video_labels)]
    return [np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in by_class])
            for _ in range(n_boot)]


def paired_delta_ci(ref_metric, var_metric, draws, level=0.95):
    """Δ = variante − referencia sobre el conjunto completo, y su IC por percentiles en los remuestreos.

    ref_metric / var_metric: funciones video_ids → métrica. Los remuestreos con métrica no definida se descartan."""
    point = var_metric(None) - ref_metric(None)
    deltas = []
    for ids in draws:
        d = var_metric(ids) - ref_metric(ids)
        if d == d:
            deltas.append(d)
    lo, hi = np.percentile(deltas, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]) if deltas else (float('nan'),) * 2
    return float(point), float(lo), float(hi)


def verdict(delta, sd_ref, sd_var, lo, hi):
    """Regla de decisión: «aporta» si |Δ| supera la desviación estándar entre semillas Y el IC 95 % excluye 0;
    «tendencia» si solo se cumple una; «no aporta» si ninguna. Se usa la mayor de las dos desviaciones
    (referencia y variante). El signo de Δ se muestra aparte."""
    sd = np.nanmax([sd_ref, sd_var]) if not (np.isnan(sd_ref) and np.isnan(sd_var)) else 0.0
    c1 = abs(delta) > sd
    c2 = (lo > 0) or (hi < 0)
    return 'aporta' if (c1 and c2) else ('tendencia' if (c1 or c2) else 'no aporta')
