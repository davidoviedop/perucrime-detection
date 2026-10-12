"""
rknn_export/scoring.py — de embeddings por fotograma a las métricas de la tesis.

Cada variante del backbone (PyTorch, ONNX, RKNN fp16, RKNN int8…) produce un embedding por
fotograma único. Aquí se arman las ventanas, se pasan por el bloque temporal + cabeza (que en
el despliegue corren en CPU) y se calculan las mismas métricas que en la etapa 2: AUC macro y
binario por ventana y por video (puntuación MIL), usando lib.evaluation.score_predictions.

Como el resto del modelo es idéntico en todas las variantes, toda diferencia de AUC entre ellas
se debe al backbone (exportación a ONNX, conversión a RKNN, cuantización).
"""
import csv
import os
from pathlib import Path

import numpy as np

from .meta import BACKBONE_IN, HEAD_IN, MEAN_01, STD_01

RESULT_COLUMNS = [
    "run_id", "backbone", "target_platform", "variant", "backbone_size_mb", "n_frames",
    "emb_cos_mean", "emb_cos_p05", "emb_relL2_mean",
    "test_window_auc_macro", "test_window_auc_binary", "test_video_auc_macro", "test_video_auc_binary",
    "test_window_accuracy", "test_video_accuracy",
    "quantized_dtype", "quantized_algorithm", "toolkit_version",
]


# ── Embeddings de referencia (PyTorch y ONNX) ─────────────────────────────────

def _normalize(frames_u8: np.ndarray) -> np.ndarray:
    """uint8 NHWC RGB -> float32 NCHW normalizado, igual que ToTensor + Normalize del entrenamiento."""
    x = frames_u8.astype(np.float32) / 255.0
    x = (x - np.asarray(MEAN_01, dtype=np.float32)) / np.asarray(STD_01, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def embeddings_torch(backbone, frames, device, batch=64) -> np.ndarray:
    import torch
    backbone = backbone.to(device).eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(frames), batch):
            x = torch.from_numpy(_normalize(np.asarray(frames[a:a + batch]))).to(device)
            out.append(backbone(x).float().cpu().numpy())
    return np.concatenate(out)


def embeddings_onnx(onnx_path, frames) -> np.ndarray:
    import onnxruntime as ort
    from tqdm.auto import tqdm
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = [sess.run(None, {BACKBONE_IN: _normalize(np.asarray(frames[i:i + 1]))})[0][0]
           for i in tqdm(range(len(frames)), desc="onnxruntime backbone", unit="frame")]
    return np.stack(out).astype(np.float32)


# ── Bloque temporal + cabeza ──────────────────────────────────────────────────

def logits_from_embeddings(emb, window_index, head_onnx) -> np.ndarray:
    """Logits (n_ventanas, n_clases) pasando cada ventana por temporal_head.onnx con onnxruntime (CPU),
    es decir, como correría en el despliegue."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(head_onnx), providers=["CPUExecutionProvider"])
    feats = emb[window_index].astype(np.float32)                    # (n_ventanas, T, D)
    return np.concatenate([sess.run(None, {HEAD_IN: feats[w:w + 1]})[0] for w in range(len(feats))])


def softmax(logits) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ── Fidelidad del embedding frente a la referencia de PyTorch ─────────────────

def embedding_fidelity(ref, emb) -> dict:
    """Similitud coseno por fotograma (media y percentil 5) y error L2 relativo medio, frente a ref."""
    num = (ref * emb).sum(axis=1)
    den = np.linalg.norm(ref, axis=1) * np.linalg.norm(emb, axis=1) + 1e-12
    cos = num / den
    rel = np.linalg.norm(ref - emb, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
    return {"emb_cos_mean": float(cos.mean()), "emb_cos_p05": float(np.percentile(cos, 5)),
            "emb_relL2_mean": float(rel.mean())}


# ── CSV de resultados ─────────────────────────────────────────────────────────

def write_rows(path, rows, run_id):
    """Agrega `rows` al CSV reemplazando las filas anteriores del mismo (run_id, variante)."""
    path = Path(path)
    new_keys = {(r["run_id"], r["variant"]) for r in rows}
    kept = []
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if (r.get("run_id"), r.get("variant")) not in new_keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)
    os.replace(tmp, path)

