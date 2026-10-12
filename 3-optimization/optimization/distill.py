"""
optimization/distill.py — destilación de conocimiento (Hinton et al.) del profesor al estudiante.

    loss = alpha * CE(estudiante, y) + (1 - alpha) * T² * KL( softmax(profesor/T) ‖ softmax(estudiante/T) )

El profesor es el modelo entrenado en la etapa 2: se carga completo (fp32, sin poda ni
cuantización), se congela y se deja en modo eval; solo se poda y cuantiza el estudiante.

Por qué el profesor corre DENTRO del forward del estudiante: run_epoch solo entrega a la
pérdida (logits, etiquetas), no el lote de entrada, así que la pérdida no puede llamar al
profesor. KDTrainWrapper corre el profesor durante el entrenamiento y deja sus logits
guardados; DistillationLoss los recoge. El profesor NO es un submódulo registrado, de modo
que parameters()/state_dict() del envoltorio son los del estudiante. Validación y test usan
el estudiante solo, y allí la pérdida es entropía cruzada simple (por eso val_loss no es
comparable con la pérdida de entrenamiento, que incluye el término KL).
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model import build_model
from lib.runs import load_run_info


def load_teacher(run_dir: Path, device: str):
    """Carga el profesor desde su corrida: (modelo congelado en eval, config, clases)."""
    run_dir = Path(run_dir)
    t_cfg, t_classes, _ = load_run_info(run_dir)
    teacher = build_model(t_cfg, len(t_classes), pretrained=False).to(device)
    teacher.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, t_cfg, t_classes


class KDTrainWrapper(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        object.__setattr__(self, '_teacher', teacher)   # evita registrarlo como submódulo
        self.teacher_logits = None

    def train(self, mode=True):
        super().train(mode)
        self._teacher.eval()   # profesor congelado: sin dropout / drop-path aunque el estudiante entrene
        return self

    def forward(self, x):
        out = self.student(x)
        if self.training:
            with torch.no_grad():
                self.teacher_logits = self._teacher(x).detach()
        return out

    def pop_teacher_logits(self):
        t, self.teacher_logits = self.teacher_logits, None
        return t


class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha, label_smoothing=0.2):
        super().__init__()
        self.T, self.alpha = temperature, alpha
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.source = None   # KDTrainWrapper; se asigna cuando ya se cargó el profesor

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        t_logits = self.source.pop_teacher_logits() if self.source is not None else None
        if t_logits is None:   # validación / sin profesor: solo la pérdida con las etiquetas
            return ce
        kd = F.kl_div(F.log_softmax(logits.float() / self.T, dim=1),
                      F.softmax(t_logits.float() / self.T, dim=1),
                      reduction='batchmean') * (self.T ** 2)
        return self.alpha * ce + (1 - self.alpha) * kd
