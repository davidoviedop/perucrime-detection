"""
optimization/compress.py — poda y cuantización post-entrenamiento de un modelo ya entrenado.

Cada combinación (cantidad de poda, tipo de dato) produce su propia copia comprimida del
modelo; el modelo original y su best.pt nunca se tocan. Dentro de cada combinación la poda
va SIEMPRE ANTES de la conversión de tipo: podar necesita la resolución real de magnitud de
fp32 para elegir los pesos más pequeños, y los módulos int8 no son nn.Linear/nn.Conv2d
simples, así que la API de poda no puede tocarlos una vez convertidos. No hay ajuste fino
después de podar, por lo que cantidades de poda muy superiores a ~0.2-0.3 cuestan precisión
real.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm.auto import tqdm


def prune_model(m, amount):
    """Poda global no estructurada por magnitud L1 sobre todos los Conv2d/Linear.

    Pone en cero los pesos de menor magnitud, en el lugar, y los fija de forma permanente
    (prune.remove): el módulo vuelve a ser un nn.Conv2d/nn.Linear simple, requisito de la
    cuantización dinámica int8. Solo pone en cero tensores fp32 densos (no hay
    almacenamiento disperso), así que por sí sola no reduce el tamaño del archivo; su valor
    es la relación precisión/cómputo.

    Devuelve (modelo, esparsidad global, filas por capa [nombre, tipo, n_pesos, n_ceros, esparsidad]).
    """
    if amount <= 0:
        return m, 0.0, []
    prunable = [(mod, 'weight') for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(prunable, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, pname in prunable:
        prune.remove(mod, pname)
    rows = [[name, type(mod).__name__, mod.weight.numel(),
             int((mod.weight == 0).sum().item()), float((mod.weight == 0).float().mean().item())]
            for name, mod in m.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    sparsity = sum(r[3] for r in rows) / sum(r[2] for r in rows)
    return m, sparsity, rows


def cast_dtype(m, dtype, device):
    """Convierte el modelo al tipo de dato pedido. Devuelve (modelo, dispositivo de evaluación, info).

    fp32: sin cambios, en `device`.
    fp16: todo el modelo a media precisión, en `device`.
    int8: se mueve a CPU (requisito del backend de cuantización dinámica) y se cuantiza
          dinámicamente solo backbone+cabeza. model.temporal (nn.TransformerEncoderLayer) se
          excluye porque la ruta rápida fusionada de su forward() lee self.linear1.weight /
          self.linear2.weight como tensores crudos, pero la cuantización dinámica reemplaza
          esas Linear por un módulo cuyo .weight es un *método*, no un tensor, y falla con
          `AttributeError: 'function' object has no attribute 'device'`.
    """
    if dtype == 'fp32':
        return m, device, {}
    if dtype == 'fp16':
        return m.half(), device, {}
    if dtype == 'int8':
        m = m.to('cpu')
        fp32_linears = {name: tuple(mod.weight.shape)
                        for name, mod in m.named_modules() if isinstance(mod, nn.Linear)}
        m.backbone = torch.ao.quantization.quantize_dynamic(m.backbone, {nn.Linear}, dtype=torch.qint8)
        m.head     = torch.ao.quantization.quantize_dynamic(m.head, {nn.Linear}, dtype=torch.qint8)
        quantized = {name: tuple(mod.weight().shape) for name, mod in m.named_modules()
                     if isinstance(mod, torch.ao.nn.quantized.dynamic.Linear)}
        return m, 'cpu', {'fp32_linears': fp32_linears, 'quantized': quantized}
    raise ValueError(f'Tipo de dato desconocido: {dtype!r}')


def predict(m, loader, dtype, device):
    """Inferencia de una variante sobre el loader (sin autocast: el tipo de dato lo da el propio modelo).

    Devuelve (probs, preds, labels) como arreglos numpy.
    """
    m.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='  eval', leave=False, unit='batch'):
            xin = x.to(device)
            if dtype == 'fp16':
                xin = xin.half()
            logits = m(xin)
            all_probs.extend(torch.softmax(logits.float(), 1).tolist())
            all_preds.extend(logits.argmax(1).tolist())
            all_labels.extend(y.tolist())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)
