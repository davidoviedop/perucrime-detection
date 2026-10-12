"""
lib/metrics.py — métricas de clasificación: AUC (macro uno-contra-resto y binaria
Normal-vs-resto), AUC por clase, y precisión / recall / F1 con matriz de confusión.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def safe_auc(y_true, y_probs, n_classes=2, normal_idx=-1):
    """Calcula el AUC binario o multiclase.

    Binario (n_classes=2): y_probs es una lista de escalares P(clase=1).
                           Devuelve (auc, nan).
    Multiclase (n_classes>2): y_probs es una lista de vectores de probabilidad.
                              Devuelve (auc_macro_ovr, auc_binario).
                              auc_binario: Normal vs resto si normal_idx >= 0.

    El AUC macro se promedia solo sobre las clases con soporte en y_true. Pasarle a
    roc_auc_score de sklearn un labels= que incluya una clase sin soporte no solo
    saca esa clase del promedio: hace que TODO el score macro sea nan (comprobado
    empíricamente), lo que rompe en silencio la selección del mejor checkpoint (se
    compara vl_auc > best_val_auc en cada época y nan siempre falla, así que nunca se
    guardaría un "mejor" checkpoint). Un split de validación pequeño o raro para una
    sola clase (p. ej. con apenas un par de videos) es un escenario real aquí.
    """
    if len(set(y_true)) < 2:
        return float('nan'), float('nan')
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_probs)), float('nan')
        probs = np.array(y_probs)
        per_class = per_class_auc(y_true, probs, n_classes)
        valid = [a for a in per_class if a == a]  # se descartan los nan (clases sin soporte)
        multi_auc = float(np.mean(valid)) if valid else float('nan')
        bin_auc = float('nan')
        if normal_idx >= 0:
            bin_true = [int(l != normal_idx) for l in y_true]
            if len(set(bin_true)) > 1:
                bin_auc = float(roc_auc_score(bin_true, 1.0 - probs[:, normal_idx]))
        return multi_auc, bin_auc
    except Exception:
        return float('nan'), float('nan')


def per_class_auc(y_true, y_probs, n_classes):
    """
    AUC-ROC uno-contra-resto de cada clase por separado (safe_auc solo da el promedio
    macro y, aparte, un único AUC binario Normal-vs-resto).

    y_probs: lista de vectores de probabilidad por clase, O (solo n_classes==2) una
    lista de escalares P(clase=1), que se normaliza internamente a un arreglo de 2
    columnas, igual que el resto del módulo.

    Devuelve una lista de largo n_classes; una clase ausente de y_true (o con muy pocos
    ejemplos para formar ambos lados de la partición OvR) recibe nan en lugar de lanzar
    una excepción, porque roc_auc_score exige que estén las dos clases.
    """
    y_true = np.asarray(y_true)
    probs  = np.asarray(y_probs, dtype=float)
    if n_classes == 2 and probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)

    out = []
    for c in range(n_classes):
        bin_true = (y_true == c).astype(int)
        if len(set(bin_true.tolist())) < 2:
            out.append(float('nan'))
            continue
        try:
            out.append(float(roc_auc_score(bin_true, probs[:, c])))
        except Exception:
            out.append(float('nan'))
    return out


def binary_auc_from_scores(labels, scores, normal_idx):
    """
    AUC-ROC de Normal-vs-resto usando puntajes de anomalía continuos arbitrarios
    (p. ej. el puntaje top-k-promedio de mil_aggregate_by_video) en vez de derivarlo de
    un vector completo de probabilidades por clase como hace safe_auc. Así el AUC
    binario refleja lo que realmente puntuó la decisión MIL, y no un valor deducido de
    un vector promedio que no tiene relación con ella.

    Devuelve nan si hay una sola clase presente (igual que la guarda de safe_auc).
    """
    bin_true = [int(l != normal_idx) for l in labels]
    if len(set(bin_true)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(bin_true, scores))
    except Exception:
        return float('nan')


def classification_metrics(y_true, y_pred, n_classes):
    """Precisión / recall / F1 (macro y por clase) y matriz de confusión.

    Devuelve un dict:
      precision_macro, recall_macro, f1_macro — floats
      precision, recall, f1                   — listas por clase (largo n_classes,
                                                 indexadas por id de clase)
      confusion_matrix                        — lista de listas (n_classes, n_classes)
      labels, preds                           — las entradas originales, como listas
                                                 (útiles para la matriz de confusión de wandb)
    """
    labels = list(range(n_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        'precision_macro': float(precision_macro),
        'recall_macro':    float(recall_macro),
        'f1_macro':        float(f1_macro),
        'precision':       precision.tolist(),
        'recall':          recall.tolist(),
        'f1':              f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'labels': list(y_true),
        'preds':  list(y_pred),
    }
