"""Fábrica de la app Flask del sitio de edición."""
from dataclasses import dataclass
from pathlib import Path

from flask import Blueprint, Flask, render_template

from lib.common.classes import TARGET_CLASSES
from lib.common.clips import ClipStore
from lib.common.paths import CLIPS_CSV, ITEMS_CSV
from lib.webkit.http import install_error_handler

from . import routes_clips, routes_raw

EDITING_DIR = Path(__file__).resolve().parent
WEBKIT_DIR  = EDITING_DIR.parent / "webkit"


@dataclass(frozen=True)
class EditingConfig:
    raw_dir:   Path          # videos fuente completos, {país}/{item_id}.{ext} (solo lectura)
    clips:     ClipStore     # clips.csv: las decisiones de edición
    items_csv: Path          # items.csv (opcional): título y URL de cada video fuente
    classes:   list[str]     # clases que se pueden asignar a un clip


def create_app(raw_dir: Path, clips_csv: Path = CLIPS_CSV, items_csv: Path = ITEMS_CSV) -> Flask:
    cfg = EditingConfig(raw_dir=Path(raw_dir).resolve(), clips=ClipStore(clips_csv),
                        items_csv=Path(items_csv), classes=list(TARGET_CLASSES))

    # Los recursos compartidos se sirven desde /static/shared/..., los propios de este sitio desde /static/editing/...
    app = Flask(__name__, template_folder=str(EDITING_DIR / "templates"),
                static_folder=str(WEBKIT_DIR / "static"), static_url_path="/static")
    app.register_blueprint(Blueprint("editing_static", __name__,
                                     static_folder=str(EDITING_DIR / "static"),
                                     static_url_path="/static/editing"))
    install_error_handler(app)

    @app.route("/")
    def index():
        return render_template("editing.html", classes=cfg.classes)

    routes_raw.register(app, cfg)
    routes_clips.register(app, cfg)
    app.config["EDITING"] = cfg
    return app
