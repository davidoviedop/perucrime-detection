"""
optimization/config.py — configuración de la destilación: la Config de la etapa 2 más los
parámetros del profesor y de la pérdida.

Los valores por defecto son los del estudiante MobileViT-XS de la corrida de referencia
(8ac40pob). Los LR son mucho más altos que los del profesor (1e-6 / 1e-5): el profesor solo
ajustaba finamente un backbone fuerte ya preentrenado, mientras que el transformer temporal
y la cabeza del estudiante parten de cero y deben aprender a imitarlo.
"""
from dataclasses import dataclass, fields

from lib.config import Config


@dataclass
class DistillConfig(Config):
    # Estudiante (pisan los valores por defecto del profesor)
    backbone: str = "mobilevit_xs"
    freeze_prefixes: tuple = ("stem", "conv_stem", "stages.0")   # se congela el stem y la primera etapa
    lr_backbone: float = 1e-4
    lr_temporal: float = 3e-4
    lr_head: float = 3e-4
    wandb_run_name: str = "distil-mobilevit-xs-mil-v2"

    # Destilación
    teacher_run_dir: str = ""          # carpeta de la corrida del profesor (outputs/{id}/ de la etapa 2)
    kd_temperature: float = 4.0        # suaviza ambas distribuciones; el factor T² mantiene comparable la escala del gradiente
    kd_alpha: float = 0.3              # peso de la pérdida con las etiquetas; (1 - alpha) va al término KL con el profesor

    def split(self):
        """(Config base, dict de destilación): lo primero es lo que entiende cualquier etapa; lo segundo va en `extra`."""
        base_names = {f.name for f in fields(Config)}
        d = self.to_dict()
        base = Config.from_dict({k: v for k, v in d.items() if k in base_names})
        extra = {k: v for k, v in d.items() if k not in base_names}
        return base, extra
