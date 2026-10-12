"""
Lógica de la etapa 3 (destilación, poda, cuantización y su comparación).

Esta etapa reutiliza el modelo, el cargador de datos, el bucle de entrenamiento y las
métricas de la etapa 2 (carpeta hermana `2-modelling/`, paquete `lib`). Al importar este
paquete se agrega esa carpeta al sys.path, por eso aquí NO se llama `lib`: sería ambiguo.
"""
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parents[1]
MODELLING_DIR = STAGE_DIR.parent / "2-modelling"

if str(MODELLING_DIR) not in sys.path:
    sys.path.insert(1, str(MODELLING_DIR))
