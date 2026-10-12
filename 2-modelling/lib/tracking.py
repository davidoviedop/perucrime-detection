"""
lib/tracking.py — registro de métricas de una corrida.

Siempre se escribe en local, sin depender de ningún servicio:
  outputs/{run_id}/history.csv     una fila por época con todas las métricas numéricas

Opcionalmente (--wandb) también se envía todo a Weights & Biases, incluidas las matrices
de confusión como gráficos. wandb es una dependencia opcional: sin ella (o sin la opción)
el entrenamiento y la evaluación funcionan igual.
"""
import csv
import json
import os
from pathlib import Path


class Tracker:
    def __init__(self, out_dir, class_names, use_wandb=False, project=None, name=None,
                 run_id=None, config=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = list(class_names)
        self.history_path = self.out_dir / "history.csv"
        self.rows: list[dict] = []
        if self.history_path.exists():                      # reanudar: se conserva lo ya registrado
            with open(self.history_path, newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))

        self.wandb = None
        self.run = None
        if use_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise SystemExit("--wandb requiere el paquete wandb (pip install wandb)") from exc
            self.wandb = wandb
            self.run = wandb.init(project=project, name=name or None, id=run_id,
                                  resume="allow", config=config)

    @property
    def url(self) -> str | None:
        return getattr(self.run, "url", None)

    def log(self, metrics: dict, plots: dict | None = None) -> None:
        """Registra un conjunto de métricas. Si trae 'epoch' se agrega una fila a history.csv.

        plots: {clave: (etiquetas, predicciones, título)}, matrices de confusión que solo
        se grafican en wandb.
        """
        if "epoch" in metrics:
            self.rows.append({k: v for k, v in metrics.items()})
            self._write_history()
        if self.wandb is not None:
            payload = dict(metrics)
            for key, (labels, preds, title) in (plots or {}).items():
                payload[key] = self.wandb.plot.confusion_matrix(
                    y_true=labels, preds=preds, class_names=self.class_names, title=title)
            self.wandb.log(payload)

    def _write_history(self) -> None:
        columns: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        tmp = self.history_path.with_name(self.history_path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, self.history_path)

    def save_json(self, filename: str, obj) -> Path:
        path = self.out_dir / filename
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))
        return path

    def log_artifact(self, path, name: str, kind: str) -> None:
        if self.wandb is not None and Path(path).exists():
            art = self.wandb.Artifact(name, type=kind)
            art.add_file(str(path))
            self.wandb.log_artifact(art)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
