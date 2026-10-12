"""
lib/evaluation.py — evaluación de un modelo sobre un split: métricas por ventana (cada
ventana contra la etiqueta de su video) y por video (puntuación MIL en dos etapas).

score_predictions() trabaja sobre arreglos de predicciones ya calculados, de modo que
la etapa 3 puede reutilizarlo para cada variante comprimida del modelo (poda, cuantización).
"""
import numpy as np
from sklearn.metrics import classification_report

from .metrics import binary_auc_from_scores
from .mil import mil_aggregate_by_video
from .scoring import scoring_dict
from .training import evaluate


def score_predictions(probs, preds, labels, recs, classes, normal_idx, cfg, split='test', tag=''):
    """Métricas por ventana y por video para un conjunto de predicciones.

    recs debe alinearse 1 a 1 con probs/preds/labels (el loader de val/test no mezcla).
    Los nombres de las métricas son {split}_window_* y {split}_video_*; con `tag` se
    insertan como {split}_window_{tag}_* (así la etapa 3 distingue cada variante).

    Devuelve un dict con:
      flat     — todas las métricas numéricas, listas para registrar (incluye accuracy y conteos)
      plots    — datos de las matrices de confusión
      window / video — las predicciones y métricas crudas de cada nivel
    """
    mid = f'{tag}_' if tag else ''
    win_prefix, vid_prefix = f'{split}_window_{mid}', f'{split}_video_{mid}'

    labels, preds = np.asarray(labels), np.asarray(preds)
    window_log, window_metrics, window_plots = scoring_dict(
        win_prefix, labels.tolist(), preds.tolist(), np.asarray(probs).tolist(),
        f'{split.capitalize()} Window Confusion Matrix' + (f' ({tag})' if tag else ''),
        classes, normal_idx)
    window_acc = float((preds == labels).mean())

    v_preds, v_probs, v_ascores, v_labels, v_paths = mil_aggregate_by_video(
        recs, np.asarray(probs).tolist(), preds.tolist(), labels.tolist(), len(classes), normal_idx,
        top_k_fraction=cfg.mil_top_k_fraction, anomaly_threshold=cfg.mil_anomaly_threshold)
    video_log, video_metrics, video_plots = scoring_dict(
        vid_prefix, v_labels, v_preds, v_probs,
        f'{split.capitalize()} Video Confusion Matrix' + (f' ({tag})' if tag else ''),
        classes, normal_idx,
        binary_auc_override=binary_auc_from_scores(v_labels, v_ascores, normal_idx))
    video_acc = float(np.mean([p == l for p, l in zip(v_preds, v_labels)])) if v_labels else float('nan')

    flat = {
        f'{win_prefix}accuracy': window_acc, f'{win_prefix}count': len(labels),
        f'{vid_prefix}accuracy': video_acc,  f'{vid_prefix}count': len(v_paths),
        **window_log, **video_log,
    }
    return {
        'flat': flat, 'plots': {**window_plots, **video_plots},
        'window': {'labels': labels, 'preds': preds, 'metrics': window_metrics, 'accuracy': window_acc,
                   'prefix': win_prefix},
        'video':  {'labels': v_labels, 'preds': v_preds, 'paths': v_paths, 'metrics': video_metrics,
                   'accuracy': video_acc, 'prefix': vid_prefix},
    }


def evaluate_split(model, loader, recs, classes, normal_idx, cfg, device, amp, split='test'):
    """Corre el modelo sobre el loader y devuelve score_predictions() de sus salidas."""
    probs, preds, labels = evaluate(model, loader, device, len(classes), amp=amp)
    return score_predictions(probs, preds, labels, recs, classes, normal_idx, cfg, split)


def print_report(res, classes, cfg):
    """Resumen por consola del resultado de score_predictions(): primero ventanas, luego videos."""
    win, vid = res['window'], res['video']
    flat = res['flat']
    wp, vp = win['prefix'], vid['prefix']

    print(f'Window AUC-ROC (macro) : {flat[wp + "auc_macro"]:.4f}')
    print(f'Window Accuracy        : {win["accuracy"]:.4f}')
    print(f'Window macro Precision/Recall/F1: '
          f'{flat[wp + "precision_macro"]:.4f} / {flat[wp + "recall_macro"]:.4f} / {flat[wp + "f1_macro"]:.4f}')
    if flat[wp + 'auc_binary'] == flat[wp + 'auc_binary']:
        print(f'Window Binary anomaly AUC (Normal vs Crime): {flat[wp + "auc_binary"]:.4f}')
    print()
    print(classification_report(win['labels'], win['preds'], target_names=classes, zero_division=0))

    print(f'\n=== Video-level scoring (MIL — top-{cfg.mil_top_k_fraction:.0%} pooling, '
          f'threshold={cfg.mil_anomaly_threshold}, {len(vid["paths"])} videos from {len(win["labels"])} windows) ===')
    print(f'Video AUC-ROC (macro) : {flat[vp + "auc_macro"]:.4f}')
    print(f'Video Accuracy        : {vid["accuracy"]:.4f}')
    print(f'Video macro Precision/Recall/F1: '
          f'{flat[vp + "precision_macro"]:.4f} / {flat[vp + "recall_macro"]:.4f} / {flat[vp + "f1_macro"]:.4f}')
    if flat[vp + 'auc_binary'] == flat[vp + 'auc_binary']:
        print(f'Video Binary anomaly AUC (Normal vs Crime): {flat[vp + "auc_binary"]:.4f}')
    print()
    print(classification_report(vid['labels'], vid['preds'], target_names=classes, zero_division=0))
