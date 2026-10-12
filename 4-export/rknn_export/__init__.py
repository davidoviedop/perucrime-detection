"""
Lógica de la etapa 4 (exportación del modelo a RKNN para NPU de Rockchip).

Esta etapa corre en DOS entornos de Python distintos, porque RKNN-Toolkit2 fija versiones
de numpy/torch/protobuf que chocan con las del proyecto:

  entorno del proyecto  (torch moderno, timm, sklearn…)   1_export_onnx.py  y  3_score.py
  entorno del toolkit   (~/venvs/rknn, solo numpy + rknn)  2_build_rknn.py

Por eso los módulos de este paquete se dividen en los que necesitan torch/timm
(onnx_export, frames, scoring) y el que solo necesita numpy + rknn (rknn_build). Importar
este paquete NO importa torch ni rknn; solo deja disponible `lib/` de la etapa 2.
"""
import sys
from pathlib import Path

STAGE_DIR     = Path(__file__).resolve().parents[1]
MODELLING_DIR = STAGE_DIR.parent / "2-modelling"
ARTIFACTS_DIR = STAGE_DIR / "artifacts"       # un subdirectorio por corrida exportada (pesado, no se versiona)
RESULTS_DIR   = STAGE_DIR / "results"          # CSV con las métricas (liviano, sí se versiona)

if str(MODELLING_DIR) not in sys.path:
    sys.path.insert(1, str(MODELLING_DIR))
