"""
lib/model.py — el detector de anomalías: backbone por fotograma + transformer temporal + cabeza.

    ventana (B, T, 3, H, W)
      └─ backbone (timm, congelado en parte) aplicado a cada fotograma → (B, T, D)
         └─ TemporalTransformer: atención entre los T fotogramas, promedio en el tiempo → (B, D)
            └─ cabeza: LayerNorm → Dropout → Linear → logits (B, n_clases)

Los nombres de los atributos (backbone._model, temporal, head) son los de los checkpoints
ya entrenados, así que no se deben cambiar sin migrar los pesos.
"""
import timm
import torch
import torch.nn as nn


class FrameBackbone(nn.Module):
    """Backbone de imágenes de timm sin clasificador, aplicado fotograma a fotograma.

    `freeze_prefixes` congela los parámetros cuyo nombre empieza con alguno de esos
    prefijos (las primeras etapas del backbone), p. ej. ('conv_stem', 'stages.0').
    """

    def __init__(self, backbone_name, pretrained=True, freeze_prefixes=()):
        super().__init__()
        self._model  = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.out_dim = self._model.num_features   # 384 en mobilevit_xs, 1024 en convnextv2_base
        for name, param in self._model.named_parameters():
            if name.startswith(tuple(freeze_prefixes)):
                param.requires_grad_(False)

    def forward(self, x):
        return self._model(x)   # (B, out_dim)


class TemporalTransformer(nn.Module):
    """Encoder de transformer sobre la secuencia temporal de características por fotograma.

    Aprende embeddings posicionales para cada posición de fotograma y aplica atención
    multi-cabeza entre los T fotogramas. La salida se promedia en el tiempo para dar una
    única representación de la secuencia.
    """

    def __init__(self, feat_dim, n_heads=4, n_layers=1, dropout=0.1, max_seq_len=16):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, feat_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads,
            dim_feedforward=feat_dim * 2,
            dropout=dropout, batch_first=True,
            norm_first=True,               # Pre-LN, por estabilidad del entrenamiento
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_embed(pos)        # suma la información de posición
        x = self.encoder(x)                # atención entre los T fotogramas
        return x.mean(dim=1)               # (B, feat_dim): promedio sobre el eje temporal


class AnomalyDetector(nn.Module):
    def __init__(self, n_classes, backbone_name, pretrained=True, freeze_prefixes=(),
                 attn_heads=4, attn_layers=1, attn_dropout=0.1, head_dropout=0.5, seq_len=16):
        super().__init__()
        self.backbone = FrameBackbone(backbone_name, pretrained, freeze_prefixes)
        feat_dim      = self.backbone.out_dim
        self.temporal = TemporalTransformer(feat_dim, attn_heads, attn_layers, attn_dropout, seq_len)
        self.head     = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.backbone(x.view(B * T, C, H, W)).view(B, T, -1)
        f = self.temporal(f)
        return self.head(f)


def build_model(cfg, n_classes, pretrained=None):
    """Crea el modelo descrito por la configuración. `pretrained` (si no es None) pisa cfg.pretrained;
    al evaluar un checkpoint se pasa False porque los pesos salen del archivo."""
    return AnomalyDetector(
        n_classes, cfg.backbone,
        pretrained=cfg.pretrained if pretrained is None else pretrained,
        freeze_prefixes=cfg.freeze_prefixes,
        attn_heads=cfg.attn_heads, attn_layers=cfg.attn_layers, attn_dropout=cfg.attn_dropout,
        head_dropout=cfg.head_dropout, seq_len=cfg.seq_len,
    )


def count_parameters(model):
    """(total, entrenables, congelados)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable
