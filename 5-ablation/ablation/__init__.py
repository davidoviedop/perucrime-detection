"""
Lógica de la etapa 5 (ablación del modelo).

Esta etapa reutiliza el modelo, el cargador de datos, el bucle de entrenamiento, la
puntuación MIL y las métricas de la etapa 2 (carpeta hermana `2-modelling/`, paquete
`lib`). Al importar este paquete se agrega esa carpeta al sys.path, por eso aquí NO se
llama `lib`: sería ambiguo.
"""
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = STAGE_DIR.parent
MODELLING_DIR = STAGE_DIR.parent / "2-modelling"
OUTPUTS_DIR = STAGE_DIR / "outputs"
RESULTS_DIR = STAGE_DIR / "results"

if str(MODELLING_DIR) not in sys.path:
    sys.path.insert(1, str(MODELLING_DIR))
if str(WORKSPACE_DIR) not in sys.path:      # run_report.py
    sys.path.insert(1, str(WORKSPACE_DIR))
