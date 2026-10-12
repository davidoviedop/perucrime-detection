"""
Clips registrados: listar, editar parámetros, reclasificar y borrar.

Todo esto modifica filas de clips.csv; ningún video se toca. Si se cambian la
ventana de recorte o el crop, la fila vuelve a `pending` y 5_render.py regenera
el video desde el crudo. Cambiar solo la clase no obliga a regenerar.
"""
from flask import jsonify, request

from lib.common.clips import to_api
from lib.webkit.http import ApiError, parse_crop, parse_trim, require_class
from lib.webkit.media import video_duration

from lib.common.rawfiles import find_raw_file


def register(app, cfg):
    @app.route("/api/stats")
    def get_stats():
        classes, items = {}, set()
        for r in cfg.clips.all():
            classes[r["class"]] = classes.get(r["class"], 0) + 1
            items.add(r["item_id"])
        return jsonify({
            "classes": classes,
            "total_clips": sum(classes.values()), "total_items": len(items),
        })

    @app.route("/api/items")
    def get_items():
        """Items que ya tienen clips registrados (vista Clips)."""
        class_f = request.args.get("class", "all")
        q       = request.args.get("q", "").strip().lower()
        min_dur = request.args.get("min_duration", type=float)
        max_dur = request.args.get("max_duration", type=float)
        filter_duration = min_dur is not None or max_dur is not None

        by_item: dict[str, dict] = {}
        for r in cfg.clips.all():
            if class_f != "all" and r["class"] != class_f:
                continue
            if q and q not in r["item_id"].lower():
                continue
            dur = float(r["end_sec"]) - float(r["start_sec"])
            it = by_item.setdefault(r["item_id"], {
                "item_id": r["item_id"], "country": r["country"], "classes": set(), "clip_count": 0,
                "max_clip_duration": None, "_matched": not filter_duration,
            })
            it["classes"].add(r["class"])
            it["clip_count"] += 1
            if filter_duration:
                it["max_clip_duration"] = max(it["max_clip_duration"] or 0.0, dur)
                if (min_dur is None or dur >= min_dur) and (max_dur is None or dur <= max_dur):
                    it["_matched"] = True

        items = [it for it in by_item.values() if it.pop("_matched")]
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
        return jsonify({
            "raw_exists": find_raw_file(cfg.raw_dir, item_id) is not None,
            "clips": [to_api(r) for r in rows],
        })

    @app.route("/api/items/<item_id>/relabel", methods=["POST"])
    def relabel_item(item_id):
        """Reclasifica de una vez todos los clips anómalos de un item (los Normal no se tocan)."""
        new_class = require_class(request.get_json(force=True), cfg.classes)
        if not cfg.clips.by_item(item_id):
            return jsonify({"error": "not found"}), 404
        return jsonify({"ok": True, "updated": cfg.clips.relabel_item(item_id, new_class)})

    @app.route("/api/clips/<clip_id>", methods=["PUT"])
    def edit_clip(clip_id):
        """
        Edita los parámetros de un clip registrado. Cuerpo de la petición (todo opcional):
          class             nueva clase.
          trim_in, trim_out nueva ventana, en la línea de tiempo del video CRUDO (van juntos).
          crop              nueva caja {left, top, right, bottom} en píxeles del crudo,
                            o null para quitar el crop (fotograma completo).
        """
        data = request.get_json(force=True)
        row = cfg.clips.get(clip_id)
        if row is None:
            return jsonify({"error": "clip not found"}), 404

        updates = {}
        if "class" in data:
            updates["class"] = require_class(data, cfg.classes)

        need_raw = ("trim_in" in data or "trim_out" in data or data.get("crop"))
        src = None
        if need_raw:
            src = find_raw_file(cfg.raw_dir, row["item_id"])
            if not src or not src.exists():
                return jsonify({"error": "raw file not found"}), 404

        if "trim_in" in data or "trim_out" in data:
            if data.get("trim_in") is None or data.get("trim_out") is None:
                raise ApiError("trim_in and trim_out go together")
            duration = video_duration(src)
            if duration <= 0:
                raise ApiError("could not read the raw video's duration", 500)
            updates["start_sec"], updates["end_sec"] = parse_trim(
                data["trim_in"], data["trim_out"], duration,
                "trim_in/trim_out must fall within the raw video's duration")

        if "crop" in data:
            updates["crop"] = parse_crop(data["crop"], src, "raw video") if data["crop"] else None

        if not updates:
            raise ApiError("nothing to edit — pass class, trim_in/trim_out and/or crop")
        return jsonify(to_api(cfg.clips.update(clip_id, **updates)))

    @app.route("/api/clips/<clip_id>", methods=["DELETE"])
    def delete_clip(clip_id):
        if not cfg.clips.delete(clip_id):
            return jsonify({"error": "clip not found"}), 404
        return jsonify({"ok": True})
