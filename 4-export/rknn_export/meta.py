"""
rknn_export/meta.py — constantes de preprocesado y el archivo export_meta.json que une los
tres pasos de la etapa (cada uno corre en un entorno distinto y solo comparte archivos).
"""
import json
from pathlib import Path

META_NAME = "export_meta.json"

# Normalización ImageNet que usa el entrenamiento (ToTensor + Normalize):
#   (pixel/255 - mean) / std   ==   (pixel - 255*mean) / (255*std)
# RKNN recibe la imagen en uint8 (0-255) y aplica la segunda forma internamente.
MEAN_01 = [0.485, 0.456, 0.406]
STD_01  = [0.229, 0.224, 0.225]
MEAN_255 = [123.675, 116.28, 103.53]
STD_255  = [58.395, 57.12, 57.375]

# Nombres de las entradas/salidas de los dos grafos ONNX
BACKBONE_IN, BACKBONE_OUT = "input", "embedding"
HEAD_IN, HEAD_OUT = "features", "logits"


def save_meta(export_dir, meta: dict) -> Path:
    path = Path(export_dir) / META_NAME
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return path


def load_meta(export_dir) -> dict:
    path = Path(export_dir) / META_NAME
    if not path.exists():
        raise SystemExit(f"{path} no existe: ejecuta primero 1_export_onnx.py")
    return json.loads(path.read_text())
