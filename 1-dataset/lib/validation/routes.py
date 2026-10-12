"""
API del sitio de validación: revisar los clips ya generados y, para cada uno,
asignarle la clase correcta o descartarlo. Nada más: la ventana de recorte y el
crop se deciden en el paso 4 (4_edit.py); aquí solo se mira el resultado.

Todo se guarda en clips.csv. Descartar no borra la fila ni el video: marca
`discarded = 1` y el clip deja de contar en 7_split.py y 8_export.py (se puede restaurar).
"""
from flask import jsonify, request

from lib.common.clips import clip_file, to_api
from lib.webkit.http import ApiError, require_class, serve_video_file


def register(app, cfg):
    def clip_dict(row: dict) -> dict:
        d = to_api(row)
        d["file_exists"] = clip_file(cfg.clips_dir, row["clip_id"]).is_file()
        if row["duration"]:                       # duración medida en el video generado
            d["duration"] = float(row["duration"])
        return d

    @app.route("/api/stats")
    def get_stats():
        classes, items = {}, set()
        discarded = pending = 0
        for r in cfg.clips.all():
            if r["discarded"] == "1":
                discarded += 1
                continue
            classes[r["class"]] = classes.get(r["class"], 0) + 1
            items.add(r["item_id"])
            pending += r["status"] != "rendered"
        return jsonify({
            "classes": classes, "total_clips": sum(classes.values()),
            "total_items": len(items), "discarded": discarded, "pending": pending,
        })

    def filtered_rows():
        """Filas que cumplen los filtros de la barra lateral (estado, clase, búsqueda, duración)."""
        status_f = request.args.get("status", "active")
        class_f  = request.args.get("class", "all")
        q        = request.args.get("q", "").strip().lower()
        min_dur  = request.args.get("min_duration", type=float)
        max_dur  = request.args.get("max_duration", type=float)
        for r in cfg.clips.all():
            if status_f == "active" and r["discarded"] == "1":
                continue
            if status_f == "discarded" and r["discarded"] != "1":
                continue
            if class_f != "all" and r["class"] != class_f:
                continue
            if q and q not in r["item_id"].lower():
                continue
            dur = clip_dict(r)["duration"]
            if (min_dur is not None and dur < min_dur) or (max_dur is not None and dur > max_dur):
                continue
            yield r, dur

    @app.route("/api/items")
    def get_items():
        filter_duration = (request.args.get("min_duration") not in (None, "")
                           or request.args.get("max_duration") not in (None, ""))
        by_item: dict[str, dict] = {}
        for r, dur in filtered_rows():
            it = by_item.setdefault(r["item_id"], {
                "item_id": r["item_id"], "country": r["country"], "classes": set(),
                "clip_count": 0, "discarded": 0, "pending": 0, "max_clip_duration": None,
            })
            it["classes"].add(r["class"])
            it["clip_count"] += 1
            it["discarded"] += r["discarded"] == "1"
            it["pending"] += r["status"] != "rendered"
            if filter_duration:
                it["max_clip_duration"] = max(it["max_clip_duration"] or 0.0, dur)
        items = list(by_item.values())
        for it in items:
            it["classes"] = sorted(it["classes"])
        if filter_duration:
            # Primero los más largos: ese es el sentido del filtro.
            items.sort(key=lambda x: (-(x["max_clip_duration"] or 0), x["item_id"]))
        else:
            items.sort(key=lambda x: x["item_id"])
        return jsonify(items)

    @app.route("/api/items/<item_id>/clips")
    def get_item_clips(item_id):
        rows = cfg.clips.by_item(item_id)
        if not rows:
            return jsonify({"error": "not found"}), 404
        return jsonify([clip_dict(r) for r in rows])

    @app.route("/api/clips/<clip_id>", methods=["PUT"])
    def edit_clip(clip_id):
        """
        Cuerpo de la petición (todo opcional):
          class      nueva clase.
          discarded  true para descartar el clip, false para restaurarlo.
        """
        data = request.get_json(force=True)
        if cfg.clips.get(clip_id) is None:
            return jsonify({"error": "clip not found"}), 404
        updates = {}
        if "class" in data:
            updates["class"] = require_class(data, cfg.classes)
        if "discarded" in data:
            updates["discarded"] = "1" if data["discarded"] else ""
        if not updates:
            raise ApiError("nothing to edit — pass class and/or discarded")
        return jsonify(clip_dict(cfg.clips.update(clip_id, **updates)))

    @app.route("/api/items/<item_id>/relabel", methods=["POST"])
    def relabel_item(item_id):
        """Reclasifica de una vez todos los clips anómalos de un item (los Normal no se tocan)."""
        new_class = require_class(request.get_json(force=True), cfg.classes)
        if not cfg.clips.by_item(item_id):
            return jsonify({"error": "not found"}), 404
        return jsonify({"ok": True, "updated": cfg.clips.relabel_item(item_id, new_class)})

    @app.route("/api/items/bulk-relabel", methods=["POST"])
    def bulk_relabel_items():
        """
        Misma semántica que relabel_item() aplicada a un lote de item_ids en una
        sola llamada. Pensado para usarse sobre la lista filtrada de la barra
        lateral: se filtra lo que haya que corregir y se reclasifica todo lo listado.

        Cuerpo: { item_ids: [...], class: "<clase válida>" }
        """
        data      = request.get_json(force=True)
        item_ids  = data.get("item_ids") or []
        new_class = require_class(data, cfg.classes)
        if not isinstance(item_ids, list) or not item_ids:
            raise ApiError("item_ids must be a non-empty list")

        wanted = set(item_ids)
        existing = {r["item_id"] for r in cfg.clips.all()} & wanted
        updated = sum(cfg.clips.relabel_item(i, new_class) for i in sorted(existing))
        return jsonify({
            "ok": True,
            "items_requested": len(item_ids),
            "items_found": len(existing),
            "items_missing": sorted(wanted - existing),
            "clips_updated": updated,
        })

    @app.route("/api/video/<clip_id>")
    def serve_video(clip_id):
        # clip_id debe ser exactamente la clave de una fila: no se puede salir de la carpeta
        if cfg.clips.get(clip_id) is None:
            return jsonify({"error": "file not found"}), 404
        path = clip_file(cfg.clips_dir, clip_id)
        if not path.is_file():
            return jsonify({"error": "file not found"}), 404
        return serve_video_file(path)
