"""Medidas de un video ya generado (para guardarlas en clips.csv)."""
from pathlib import Path

import cv2

from lib.webkit.media import video_duration


def probe_video(path: Path) -> dict:
    """n_frames, fps, width, height y duration (s) del video, o {} si no se puede abrir.

    n_frames es el conteo de fotogramas del contenedor (CAP_PROP_FRAME_COUNT), el
    mismo que usa la división para calcular las ventanas de entrenamiento."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {}
        return {
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps":      round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
            "width":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": round(video_duration(path), 3),
        }
    finally:
        cap.release()
