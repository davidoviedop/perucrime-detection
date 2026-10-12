"""Fábrica de la app Flask del sitio de validación."""
from dataclasses import dataclass
from pathlib import Path

from flask import Blueprint, Flask, render_template

from lib.common.classes import TARGET_CLASSES
from lib.common.clips import ClipStore
from lib.common.paths import CLIPS_CSV, CLIPS_DIR
from lib.webkit.http import install_error_handler

from . import routes

VALIDATION_DIR = Path(__file__).resolve().parent
WEBKIT_DIR     = VALIDATION_DIR.parent / "webkit"


@dataclass(frozen=True)
class ValidationConfig:
    clips:     ClipStore     # clips.csv: las decisiones de edición y de validación
    clips_dir: Path          # videos generados por 5_render.py, {clip_id}.mp4
    classes:   list[str]     # clases que se pueden asignar a un clip


def create_app(clips_csv: Path = CLIPS_CSV, clips_dir: Path = CLIPS_DIR) -> Flask:
    cfg = ValidationConfig(clips=ClipStore(clips_csv), clips_dir=Path(clips_dir).resolve(),
                           classes=list(TARGET_CLASSES))

    # Los recursos compartidos se sirven desde /static/shared/..., los propios de este sitio desde /static/validation/...
    app = Flask(__name__, template_folder=str(VALIDATION_DIR / "templates"),
                static_folder=str(WEBKIT_DIR / "static"), static_url_path="/static")
    app.register_blueprint(Blueprint("validation_static", __name__,
                                     static_folder=str(VALIDATION_DIR / "static"),
                                     static_url_path="/static/validation"))
    install_error_handler(app)

    @app.route("/")
    def index():
        return render_template("validation.html", classes=cfg.classes)

    routes.register(app, cfg)
    app.config["VALIDATION"] = cfg
    return app
