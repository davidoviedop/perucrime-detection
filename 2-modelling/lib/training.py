"""
lib/training.py — bucle de entrenamiento y evaluación a nivel de ventana, con
checkpoints, reanudación, scheduler ReduceLROnPlateau y parada temprana.

Todo el entrenamiento es a nivel de VENTANA (cada ventana hereda la etiqueta de su
video; aceptado, porque los clips pueden ser largos con el incidente en solo una parte).
La puntuación MIL a nivel de video (lib/mil.py) solo cambia cómo se REPORTAN las
métricas de validación y de test; no influye en el entrenamiento ni en qué
checkpoint se elige como mejor (eso sale del AUC por ventana).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm, trange

from .metrics import binary_auc_from_scores, classification_metrics, safe_auc
from .mil import mil_aggregate_by_video
from .scoring import scoring_dict


# ── Una época y la inferencia ─────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, scheduler=None,
              device='cpu', train=True, n_classes=2, normal_idx=-1,
              clip_grad=1.0, desc='', scheduler_step='epoch',
              amp=False, scaler=None):
    """Época genérica de entrenamiento o evaluación.

    Devuelve (loss_promedio, auc, bin_auc, metrics).
      Binario    : auc = AUC-ROC, bin_auc = nan.
      Multiclase : auc = AUC macro OvR, bin_auc = AUC binario de anomalía.
      metrics    : dict de classification_metrics() (precisión/recall/F1 macro y por
                   clase, matriz de confusión) más 'probs' (probabilidades crudas por
                   ventana, para mil_aggregate_by_video()).

    scheduler_step:
      'batch' — se llama a step() tras cada paso del optimizador (OneCycleLR).
      'epoch' — aquí no se llama; quien llama hace el step al volver.

    amp / scaler:
      amp=True ejecuta el forward (y la pérdida) bajo autocast, lo que reduce a la mitad
      la memoria de activaciones en CUDA. scaler es el GradScaler que des-escala los
      gradientes antes del paso del optimizador; es obligatorio si amp=True y train=True
      (el backward en fp16 solo hace underflow de los gradientes pequeños).
    """
    model.train() if train else model.eval()
    tot_loss, all_probs, all_labels, all_preds = 0.0, [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
    with ctx:
        for x, y in tqdm(loader, desc=f'  {desc}', leave=False,
                         unit='batch', disable=not desc):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            with torch.autocast(device_type=device_type, enabled=amp):
                logits = model(x)
                loss   = criterion(logits, y)
            if train:
                if amp:
                    scaler.scale(loss).backward()
                    if clip_grad:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                if scheduler is not None and scheduler_step == 'batch':
                    scheduler.step()
            tot_loss += loss.item() * len(y)
            # Softmax en fp32 sin importar amp: las filas en fp16 no suman exactamente
            # 1.0, lo que el roc_auc_score multiclase de sklearn rechaza.
            if n_classes == 2:
                all_probs.extend(
                    torch.softmax(logits.float(), 1)[:, 1].detach().cpu().tolist())
            else:
                all_probs.extend(
                    torch.softmax(logits.float(), 1).detach().cpu().tolist())
            all_preds.extend(logits.argmax(1).detach().cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = tot_loss / max(len(all_labels), 1)
    auc, bin_auc = safe_auc(all_labels, all_probs, n_classes, normal_idx)
    metrics = classification_metrics(all_labels, all_preds, n_classes)
    metrics['probs'] = all_probs
    return avg_loss, auc, bin_auc, metrics


def evaluate(model, loader, device, n_classes=2, amp=False):
    """Inferencia sobre el loader.

    Devuelve (probs, preds, labels) como arreglos numpy.
      Binario    : probs de forma (N,).
      Multiclase : probs de forma (N, n_classes).
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
    with torch.no_grad():
        for x, y in tqdm(loader, desc='  eval', leave=False, unit='batch'):
            with torch.autocast(device_type=device_type, enabled=amp):
                logits = model(x.to(device))
            if n_classes == 2:
                all_probs.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().tolist())
            else:
                all_probs.extend(torch.softmax(logits.float(), 1).cpu().tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


# ── Optimizador y scheduler ───────────────────────────────────────────────────

def build_optimizer(model, cfg):
    """AdamW con tres grupos de parámetros (backbone entrenable, transformer temporal, cabeza)."""
    return optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg.lr_backbone},
        {'params': model.temporal.parameters(), 'lr': cfg.lr_temporal},
        {'params': model.head.parameters(),     'lr': cfg.lr_head},
    ], weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=cfg.lr_factor, patience=cfg.lr_sched_patience,
        threshold=cfg.lr_threshold, min_lr=cfg.lr_min,
    )


# ── Checkpoints ───────────────────────────────────────────────────────────────

_HISTORY_KEYS = ('train_loss', 'val_loss', 'train_auc', 'val_auc', 'train_bin_auc', 'val_bin_auc',
                 'train_precision_macro', 'val_precision_macro',
                 'train_recall_macro', 'val_recall_macro',
                 'train_f1_macro', 'val_f1_macro')


def empty_history():
    return {k: [] for k in _HISTORY_KEYS}


def restore_checkpoint(path, model, optimizer, scheduler, scaler, amp, device):
    """Reanuda desde un checkpoint de última época.

    Devuelve un dict {start_epoch, best_val_auc, best_state, history} o None si el
    checkpoint es incompatible con el modelo actual (en ese caso se borra y se empieza
    de cero).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if amp and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        history = ckpt['history']
        for key in _HISTORY_KEYS:
            if key not in history:
                history[key] = [float('nan')] * len(history['train_loss'])
        print(f'Resumed from checkpoint: epoch {ckpt["epoch"]}, best_val_auc={ckpt["best_val_auc"]:.4f}')
        return {'start_epoch': ckpt['epoch'] + 1, 'best_val_auc': ckpt['best_val_auc'],
                'best_state': ckpt['best_state'], 'history': history}
    except Exception as exc:
        print(f'Checkpoint incompatible ({type(exc).__name__}: {exc}) — '
              f'deleting stale checkpoint and starting from scratch.')
        Path(path).unlink()
        return None


# ── Entrenamiento completo ────────────────────────────────────────────────────

def fit(model, train_loader, val_loader, vl_recs, criterion, optimizer, scheduler, scaler,
        cfg, classes, normal_idx, device, amp, tracker, checkpoint_path, run_id,
        resume=None, train_model=None):
    """Entrena `model` de la época `start_epoch` a `cfg.epochs` (o hasta parar antes).

    train_model: módulo que se usa en la época de entrenamiento si no es el propio
    `model` (la destilación de la etapa 3 pasa aquí un envoltorio que además corre al
    profesor). La validación y los checkpoints usan siempre `model`.
    resume: dict de restore_checkpoint() para continuar una corrida.

    Cada época: entrena, valida (ventana y video MIL), hace step del scheduler sobre
    val_auc, registra, guarda el checkpoint y aplica la parada temprana. El mejor modelo
    es el de mayor val_auc a nivel de ventana. Devuelve (best_val_auc, best_state, history).
    """
    train_model = train_model if train_model is not None else model
    n_classes = len(classes)
    start_epoch  = resume['start_epoch'] if resume else 1
    best_val_auc = resume['best_val_auc'] if resume else -1.0
    best_state   = resume['best_state'] if resume else None
    history      = resume['history'] if resume else empty_history()

    print(f'Training epochs {start_epoch}–{cfg.epochs} on {device}  '
          f'(early-stop patience={cfg.patience}) ...')
    patience_counter = 0
    for epoch in trange(start_epoch, cfg.epochs + 1, desc='Epochs', unit='ep'):
        tr_loss, tr_auc, tr_bin_auc, tr_metrics = run_epoch(
            train_model, train_loader, criterion, optimizer, None,
            device, train=True, n_classes=n_classes, normal_idx=normal_idx,
            desc='train', amp=amp, scaler=scaler)
        vl_loss, vl_auc, vl_bin_auc, vl_metrics = run_epoch(
            model, val_loader, criterion,
            device=device, train=False, n_classes=n_classes, normal_idx=normal_idx,
            desc='val  ', amp=amp)

        # ReduceLROnPlateau hace un step por época sobre la métrica vigilada. Tanto esto
        # como la selección del mejor checkpoint siguen a nivel de ventana (vl_auc,
        # directo de run_epoch): el entrenamiento no depende de la puntuación MIL a nivel
        # de video que se calcula abajo, que es solo para reportar. El scheduler y la
        # parada temprana vigilan la MISMA métrica (val_auc, maximizar); sus paciencias
        # son cfg.lr_sched_patience y cfg.patience.
        scheduler.step(vl_auc)

        # val_loader tiene shuffle=False, así que vl_metrics['probs']/['preds']/['labels']
        # calzan 1 a 1 con vl_recs; esto NO valdría para el loader de entrenamiento
        # (shuffle=True con reemplazo), por eso nunca se calcula puntuación de video para
        # tr_metrics.
        vv_preds, vv_probs, vv_ascores, vv_labels, _ = mil_aggregate_by_video(
            vl_recs, vl_metrics['probs'], vl_metrics['preds'], vl_metrics['labels'],
            n_classes, normal_idx,
            top_k_fraction=cfg.mil_top_k_fraction, anomaly_threshold=cfg.mil_anomaly_threshold)

        val_window_log, _, val_window_plots = scoring_dict(
            'val_window_', vl_metrics['labels'], vl_metrics['preds'], vl_metrics['probs'],
            'Val Window Confusion Matrix', classes, normal_idx)
        val_video_log, _, val_video_plots = scoring_dict(
            'val_video_', vv_labels, vv_preds, vv_probs,
            'Val Video Confusion Matrix', classes, normal_idx,
            binary_auc_override=binary_auc_from_scores(vv_labels, vv_ascores, normal_idx))
        val_video_acc = float(np.mean([p == l for p, l in zip(vv_preds, vv_labels)])) if vv_labels else float('nan')

        history['train_loss'].append(tr_loss);       history['val_loss'].append(vl_loss)
        history['train_auc'].append(tr_auc);         history['val_auc'].append(vl_auc)
        history['train_bin_auc'].append(tr_bin_auc); history['val_bin_auc'].append(vl_bin_auc)
        history['train_precision_macro'].append(tr_metrics['precision_macro'])
        history['val_precision_macro'].append(vl_metrics['precision_macro'])
        history['train_recall_macro'].append(tr_metrics['recall_macro'])
        history['val_recall_macro'].append(vl_metrics['recall_macro'])
        history['train_f1_macro'].append(tr_metrics['f1_macro'])
        history['val_f1_macro'].append(vl_metrics['f1_macro'])

        lr_backbone, lr_temporal, lr_head = (g['lr'] for g in optimizer.param_groups)
        epoch_log = {
            'epoch': epoch,
            'train_loss': tr_loss,        'val_loss': vl_loss,
            'train_auc':  tr_auc,         'val_auc':  vl_auc,
            'train_bin_auc': tr_bin_auc,  'val_bin_auc': vl_bin_auc,
            'val_window_count': len(vl_metrics['labels']),
            'val_video_count':  len(vv_labels),
            'val_video_accuracy': val_video_acc,
            'train_precision_macro': tr_metrics['precision_macro'],
            'train_recall_macro':    tr_metrics['recall_macro'],
            'train_f1_macro':        tr_metrics['f1_macro'],
            'lr_backbone': lr_backbone, 'lr_temporal': lr_temporal, 'lr_head': lr_head,
        }
        epoch_log.update(val_window_log)
        epoch_log.update(val_video_log)
        for i, cls in enumerate(classes):
            epoch_log[f'train_precision_{cls}'] = tr_metrics['precision'][i]
            epoch_log[f'train_recall_{cls}']    = tr_metrics['recall'][i]
            epoch_log[f'train_f1_{cls}']        = tr_metrics['f1'][i]
        plots = {**val_window_plots, **val_video_plots,
                 'train_confusion_matrix': (tr_metrics['labels'], tr_metrics['preds'], 'Train Confusion Matrix')}
        tracker.log(epoch_log, plots)

        if not (vl_auc != vl_auc) and vl_auc > best_val_auc:
            best_val_auc     = vl_auc
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_auc': best_val_auc, 'best_state': best_state,
            'history': history, 'run_id': run_id,
        }, checkpoint_path)

        print(f'Ep {epoch:02d}/{cfg.epochs}  '
              f'tr_loss={tr_loss:.4f}  tr_auc={tr_auc:.4f}  tr_bin={tr_bin_auc:.4f}  |  '
              f'vl_loss={vl_loss:.4f}  vl_auc={vl_auc:.4f}  vl_bin={vl_bin_auc:.4f}  '
              f'vl_video_auc={val_video_log["val_video_auc_macro"]:.4f} (n={len(vv_labels)})  '
              f'vl_prec={vl_metrics["precision_macro"]:.4f}  '
              f'vl_rec={vl_metrics["recall_macro"]:.4f}  '
              f'vl_f1={vl_metrics["f1_macro"]:.4f}  '
              f'lr_head={lr_head:.2e}  '
              f'[patience {patience_counter}/{cfg.patience}]')

        if patience_counter >= cfg.patience:
            print(f'Early stopping — val_auc not improved for {cfg.patience} epochs.')
            break

    return best_val_auc, best_state, history
