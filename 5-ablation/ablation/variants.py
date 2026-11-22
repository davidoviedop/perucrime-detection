"""
ablation/variants.py — variantes de arquitectura para la ablación.

El modelo de referencia (lib/model.py: AnomalyDetector) es

    backbone por fotograma ─► TemporalTransformer ─► cabeza (LayerNorm · Dropout · Linear)

Cada variante cambia UNA pieza y deja el resto igual:

  transformer         módulo temporal de la referencia (embedding posicional + atención)
  transformer_nopos   el mismo transformer sin embedding posicional: no distingue el orden
  mean                sin módulo temporal: promedio de las características de los fotogramas
  lstm                LSTM de 2 capas y 256 unidades (el módulo recurrente clásico)

Las variantes que no son de arquitectura (congelar, label smoothing, jitter, ventana…) son
campos de Config y se declaran en el YAML de la ablación; el muestreador se cambia con un
gancho de los loaders (suite.py).
"""
import torch.nn as nn

from lib.model import build_model

TEMPORAL_KINDS = ('transformer', 'transformer_nopos', 'mean', 'lstm')


class TemporalMean(nn.Module):
    """Sin módulo temporal: promedia las características de los T fotogramas."""
    def forward(self, x):                      # (B, T, D) → (B, D)
        return x.mean(dim=1)


class TemporalTransformerNoPos(nn.Module):
    """El encoder de la referencia pero sin embedding posicional: la atención es invariante
    al orden de los fotogramas (equivale a tratar la ventana como un conjunto)."""
    def __init__(self, feat_dim, n_heads=4, n_layers=1, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=feat_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x).mean(dim=1)


class TemporalLSTM(nn.Module):
    """LSTM sobre los T fotogramas; se usa la última salida (como en la línea base CNN+LSTM)."""
    def __init__(self, feat_dim, hidden=256, layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1]


def build_ablation_model(cfg, n_classes, pretrained=None, temporal='transformer', lstm_hidden=256):
    """Modelo de la referencia con el módulo temporal indicado.

    Usa el mismo build_model de la etapa 2 y, si `temporal` no es 'transformer', reemplaza
    `.temporal` (y la cabeza si cambia la dimensión). Los nombres de atributos se conservan
    (backbone, temporal, head) para que el optimizador y los checkpoints funcionen igual.
    """
    if temporal not in TEMPORAL_KINDS:
        raise ValueError(f'temporal desconocido: {temporal!r} (válidos: {TEMPORAL_KINDS})')
    model = build_model(cfg, n_classes, pretrained=pretrained)
    if temporal == 'transformer':
        return model
    feat_dim = model.backbone.out_dim
    if temporal == 'mean':
        model.temporal = TemporalMean()
    elif temporal == 'transformer_nopos':
        model.temporal = TemporalTransformerNoPos(feat_dim, cfg.attn_heads, cfg.attn_layers, cfg.attn_dropout)
    elif temporal == 'lstm':
        model.temporal = TemporalLSTM(feat_dim, lstm_hidden, dropout=cfg.head_dropout)
        out_dim = model.temporal.out_dim
        model.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(cfg.head_dropout),
                                   nn.Linear(out_dim, n_classes))
    return model
