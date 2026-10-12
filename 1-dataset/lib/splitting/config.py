"""Constantes de la división balanceada por cantidad de ventanas."""

SOURCES      = ["scrapped", "ucf-crime", "vioperu"]
SPLIT_NAMES  = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
FOLD    = {"Road": "Road_Accident"}   # nombre de carpeta antiguo -> clase canónica
EXCLUDE = {"Others"}                  # cajón de material sin clase, nunca se divide (ver lib/common/classes.py)

# Deben coincidir exactamente con SEQ_LEN / STRIDE / FRAME_STRIDE /
# MAX_WINDOWS_PER_VIDEO de la etapa de Modelado: estos números SON la definición
# de "ventana" para la división. (Una ventana abarca (16-1)*4+1 = 61 fotogramas fuente.)
SEQ_LEN      = 16
STRIDE       = 16
FRAME_STRIDE = 4
MAX_WINDOWS_PER_VIDEO = 70
SPAN = (SEQ_LEN - 1) * FRAME_STRIDE + 1

# 'scrapped' sale de clips.csv (lo que se editó a mano); las demás fuentes son
# datasets externos que se leen de carpetas sueltas: data/external/{fuente}/{clase}/
LOCAL_SOURCE = "scrapped"

DEFAULT_RESTARTS = 60
