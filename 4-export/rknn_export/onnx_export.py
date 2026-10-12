"""
rknn_export/onnx_export.py — exporta el modelo a DOS grafos ONNX con forma estática.

    backbone.onnx       (1, 3, H, W)  ->  (1, D)       backbone por fotograma; va a la NPU
    temporal_head.onnx  (1, T, D)     ->  (1, N)       transformer temporal + cabeza; va a la CPU

La partición se hace aquí porque el backbone es el 99.7 % del cómputo, mientras que el
bloque temporal solo ve 16 embeddings de D valores (no píxeles). Todas las formas son
fijas (lote 1) a propósito: RKNN trabaja con formas estáticas y varios operadores
(Softmax, Slice) solo admiten lote 1.

El grafo del backbone recibe la imagen YA NORMALIZADA, igual que en PyTorch. Quien la
normaliza en el despliegue es RKNN (mean_values / std_values de rknn.config), de modo
que a la NPU se le entrega la imagen cruda en uint8.
"""
import numpy as np
import torch
import torch.nn as nn

from .meta import BACKBONE_IN, BACKBONE_OUT, HEAD_IN, HEAD_OUT


class BackboneWrapper(nn.Module):
    """(1, 3, H, W) normalizado -> (1, D)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class TemporalHeadWrapper(nn.Module):
    """(1, T, D) -> (1, n_clases): transformer temporal + cabeza, igual que AnomalyDetector.forward."""

    def __init__(self, model):
        super().__init__()
        self.temporal = model.temporal
        self.head = model.head

    def forward(self, f):
        return self.head(self.temporal(f))


def _export(module, example, path, input_name, output_name, opset):
    module.eval()
    # La ruta rápida fusionada de nn.TransformerEncoder (inferencia) no se exporta a ONNX:
    # se desactiva para que salga el grafo normal (MatMul + Softmax).
    mha = getattr(torch.backends, "mha", None)
    if mha is not None and hasattr(mha, "set_fastpath_enabled"):
        mha.set_fastpath_enabled(False)

    kwargs = dict(input_names=[input_name], output_names=[output_name],
                  opset_version=opset, do_constant_folding=True)
    with torch.no_grad():
        try:
            # Exportador clásico (TorchScript): el más predecible para RKNN.
            torch.onnx.export(module, (example,), str(path), dynamo=False, **kwargs)
        except Exception as legacy_err:
            try:
                # Si esta versión de torch ya no lo ofrece, el exportador por defecto (requiere onnxscript).
                torch.onnx.export(module, (example,), str(path), **kwargs)
            except Exception as err:
                raise RuntimeError(
                    f"No se pudo exportar {path.name}: exportador clásico -> {legacy_err!r}; "
                    f"exportador por defecto -> {err!r}") from err


def max_abs_diff(module, example, path, input_name) -> float:
    """Diferencia máxima entre PyTorch y el ONNX exportado (onnxruntime en CPU) sobre la misma entrada."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = module(example).cpu().numpy()
    out = sess.run(None, {input_name: example.cpu().numpy()})[0]
    return float(np.abs(ref - out).max())


def export_backbone(model, path, img_size, opset=17):
    """Exporta el backbone y devuelve la diferencia máxima respecto a PyTorch."""
    wrapper = BackboneWrapper(model.backbone).cpu().float().eval()
    example = torch.randn(1, 3, *img_size)
    _export(wrapper, example, path, BACKBONE_IN, BACKBONE_OUT, opset)
    return max_abs_diff(wrapper, example, path, BACKBONE_IN)


def export_temporal_head(model, path, seq_len, feat_dim, opset=17):
    """Exporta el bloque temporal + cabeza y devuelve la diferencia máxima respecto a PyTorch."""
    wrapper = TemporalHeadWrapper(model).cpu().float().eval()
    example = torch.randn(1, seq_len, feat_dim)
    _export(wrapper, example, path, HEAD_IN, HEAD_OUT, opset)
    return max_abs_diff(wrapper, example, path, HEAD_IN)
