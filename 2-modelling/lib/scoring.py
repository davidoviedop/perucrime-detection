"""
lib/scoring.py — arma el conjunto de métricas de un (split, nivel).

Una sola función para que el nivel de ventana y el de video (en validación, en cada
época, y en la pasada final de test) reporten siempre el mismo conjunto de métricas bajo
un prefijo uniforme {split}_{nivel}_, p. ej. val_window_auc_macro o test_video_auc_Robbery.
"""
from .metrics import classification_metrics, per_class_auc, safe_auc


def scoring_dict(prefix, labels, preds, probs, title, class_names, normal_idx,
                 binary_auc_override=None):
    """
    Construye un dict plano (apto para registrar) para una combinación (split, nivel).

    Incluye AUC, precisión, recall y F1 macro y por clase (el AUC por clase es
    uno-contra-resto; ver per_class_auc), más los datos de una matriz de confusión
    titulada `title`.

    Devuelve (flat, metrics, plots):
      flat     — dict de números {prefijo + nombre: valor}.
      metrics  — dict crudo de classification_metrics (etiquetas, predicciones, matriz...).
      plots    — {prefijo + 'confusion_matrix': (etiquetas, predicciones, título)};
                 el Tracker lo convierte en el gráfico de wandb si está activo.

    binary_auc_override: si se indica, reemplaza al AUC binario que deriva safe_auc del
    vector de probabilidad medio por clase. La puntuación MIL a nivel de video pasa aquí
    el puntaje top-k-promedio de mil_aggregate_by_video: una señal continua más fiel
    para el AUC Normal-vs-resto, porque es el puntaje que realmente umbralizó la
    decisión en dos etapas.
    """
    n_classes = len(class_names)
    auc_macro, auc_binary = safe_auc(labels, probs, n_classes, normal_idx)
    if binary_auc_override is not None:
        auc_binary = binary_auc_override
    auc_per_class = per_class_auc(labels, probs, n_classes)
    metrics = classification_metrics(labels, preds, n_classes)

    flat = {
        f'{prefix}auc_macro':       auc_macro,
        f'{prefix}auc_binary':      auc_binary,
        f'{prefix}precision_macro': metrics['precision_macro'],
        f'{prefix}recall_macro':    metrics['recall_macro'],
        f'{prefix}f1_macro':        metrics['f1_macro'],
    }
    for i, cls in enumerate(class_names):
        flat[f'{prefix}auc_{cls}']       = auc_per_class[i]
        flat[f'{prefix}precision_{cls}'] = metrics['precision'][i]
        flat[f'{prefix}recall_{cls}']    = metrics['recall'][i]
        flat[f'{prefix}f1_{cls}']        = metrics['f1'][i]
    plots = {f'{prefix}confusion_matrix': (metrics['labels'], metrics['preds'], title)}
    return flat, metrics, plots
