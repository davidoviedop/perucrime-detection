"""
lib/config.py — hiperparámetros de un entrenamiento, con carga desde YAML.

Los valores por defecto son los de la corrida del profesor (ConvNeXtV2-base). Cada
corrida real se describe con un YAML en config/ (los valores exactos que registró
wandb) y se puede sobrescribir cualquier campo desde la línea de comandos.

Toda corrida guarda su configuración completa en outputs/{run_id}/run.yaml, que es lo
que lee 2_evaluate.py (y la etapa 3) para reconstruir el mismo modelo.
"""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # ── Datos ───────────────────────────────────────────────────────────────
    data_root: str = "../1-dataset/data/3_dataset"   # {fuente}/{split}/{clase}/*.mp4 (salida de 1-dataset)
    seq_len: int = 16                  # fotogramas por ventana
    stride: int = 16                   # paso entre inicios de ventana (fotogramas fuente)
    frame_stride: int = 4              # paso entre fotogramas muestreados dentro de una ventana
    img_size: tuple = (160, 160)
    max_windows_per_video: int = 70    # tope de ventanas por video
    # Aumentación temporal (solo train): en cada lectura, el inicio de la ventana se desplaza
    # un número aleatorio de fotogramas en [0, temporal_jitter]. 0 = desactivado (como las
    # corridas de referencia). Valor recomendado: stride - 1 (= 15), que cubre todo el hueco
    # entre dos inicios de ventana sin cambiar la cantidad de ventanas ni el split.
    temporal_jitter: int = 0
    batch_size: int = 16
    num_workers: int = 4

    # ── Modelo ──────────────────────────────────────────────────────────────
    backbone: str = "convnextv2_base"  # nombre del modelo en timm
    pretrained: bool = True            # pesos de ImageNet al inicializar el backbone
    freeze_prefixes: tuple = ("conv_stem", "stages.0", "stages.1")   # parámetros del backbone congelados
    attn_heads: int = 4
    attn_layers: int = 1
    attn_dropout: float = 0.1
    head_dropout: float = 0.5

    # ── Entrenamiento ───────────────────────────────────────────────────────
    epochs: int = 2
    patience: int = 10                 # parada temprana sobre val_auc
    lr_backbone: float = 1e-6
    lr_temporal: float = 1e-5
    lr_head: float = 1e-5
    weight_decay: float = 1e-3
    label_smoothing: float = 0.2
    # ReduceLROnPlateau sobre val_auc (maximizar): baja el LR de todos los grupos por
    # lr_factor tras lr_sched_patience épocas sin subir al menos lr_threshold. Más corta
    # que `patience`, para que el LR pueda bajar (quizá varias veces) antes de parar.
    lr_factor: float = 0.5
    lr_sched_patience: int = 3
    lr_threshold: float = 1e-4
    lr_min: float = 1e-8
    seed: int = 42

    # ── Puntuación MIL por video (solo afecta a las métricas reportadas, no al entrenamiento) ──
    mil_top_k_fraction: float = 0.10   # fracción de ventanas de un video que se agrupan para decidir si es anómalo
    mil_anomaly_threshold: float = 0.50   # P(no Normal) promedio del top-k a partir de la cual el video es anómalo

    # ── Seguimiento ─────────────────────────────────────────────────────────
    wandb_project: str = "peru-crime4-anomaly"
    wandb_run_name: str = ""

    # ── Utilidades ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        d["img_size"] = list(self.img_size)
        d["freeze_prefixes"] = list(self.freeze_prefixes)
        return d

    @classmethod
    def from_dict(cls, data: dict):
        known = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"campos desconocidos en la configuración: {sorted(unknown)}")
        kwargs = dict(data)
        for key in ("img_size", "freeze_prefixes"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path, **overrides):
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(data)

    def save(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))
