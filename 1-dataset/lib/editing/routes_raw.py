"""
Vista Raw: navegar los videos fuente y REGISTRAR clips sobre ellos.

Aquí no se codifica ningún video: "Guardar clip" solo anota en clips.csv de qué
video crudo, en qué ventana de tiempo, con qué crop y en qué clase sale el clip.
El video lo genera después 5_render.py. El archivo fuente nunca se abre para
escritura: es la única copia de ese material.
"""
from flask import jsonify, request

from lib.common.clips import to_api
from lib.webkit.durations import cached_durations
from lib.webkit.http import ApiError, parse_crop, parse_trim, require_class, serve_video_file
from lib.webkit.media import video_duration

from lib.common.rawfiles import find_raw_file, iter_raw_files


def register(app, cfg):
    @app.route("/api/raw-stats")
    def get_raw_stats():
        countries = {}
        total = 0
        for country, item_id, path in iter_raw_files(cfg.raw_dir):
            countries[country] = countries.get(country, 0) + 1
            total += 1
        clips = cfg.clips.all()
        return jsonify({
            "countries": countries,
            "total_raw": total,
            "clips_recorded": len(clips),
            "items_with_clips": len({c["item_id"] for c in clips}),
        })

    @app.route("/api/raw-items")
    def get_raw_items():
        country_f = request.args.get("country", "all")
        q         = request.args.get("q", "").strip().lower()
        min_dur   = request.args.get("min_duration", type=float)
        max_dur   = request.args.get("max_duration", type=float)

        candidates = []
        for country, item_id, path in iter_raw_files(cfg.raw_dir):
            if country_f != "all" and country != country_f:
                continue
            if q and q not in item_id.lower():
                continue
            candidates.append((country, item_id, path))

        durations = cached_durations([c[2] for c in candidates])
        counts: dict[str, int] = {}
        for c in cfg.clips.all():
            counts[c["item_id"]] = counts.get(c["item_id"], 0) + 1

        items = []
        for country, item_id, path in candidates:
            dur = durations.get(path, 0.0)
            if min_dur is not None and dur < min_dur:
                continue
            if max_dur is not None and dur > max_dur:
                continue
            items.append({
                "item_id":    item_id,
                "country":    country,
                "duration":   round(dur, 2),
                "clip_count": counts.get(item_id, 0),
            })
        items.sort(key=lambda x: x["item_id"])
        return jsonify(items)

    @app.route("/api/raw-video/<item_id>")
    def serve_raw_video(item_id):
        path = find_raw_file(cfg.raw_dir, item_id)
        if not path or not path.exists():
            return jsonify({"error": "raw file not found"}), 404
        return serve_video_file(path)

    @app.route("/api/raw-items/<item_id>/clips", methods=["POST"])
    def record_clip(item_id):
        """
        Registra un clip nuevo sobre un video crudo. Cuerpo de la petición:
          class             OBLIGATORIO, una de las clases válidas.
          trim_in, trim_out opcionales, segundos en la línea de tiempo del propio
                            video crudo; por defecto el video completo (0..duración).
          crop              opcional {left, top, right, bottom} en píxeles del
                            video crudo; por defecto el fotograma completo.
        No codifica nada: solo agrega una fila `pending` a clips.csv.
        """
        data = request.get_json(force=True)
        new_class = require_class(data, cfg.classes)

        src = find_raw_file(cfg.raw_dir, item_id)
        if not src or not src.exists():
            return jsonify({"error": "raw file not found"}), 404

        duration = video_duration(src)
        if duration <= 0:
            raise ApiError("could not read the raw video's duration", 500)

        trim_in  = data.get("trim_in")
        trim_out = data.get("trim_out")
        trim_in, trim_out = parse_trim(
            0.0 if trim_in is None else trim_in,
            duration if trim_out is None else trim_out,
            duration, "trim_in/trim_out must fall within the raw video's duration")

        crop = data.get("crop")
        if crop:
            crop = parse_crop(crop, src, "raw video")

        row = cfg.clips.add(
            item_id=item_id, country=src.parent.name,
            raw_file=src.relative_to(cfg.raw_dir).as_posix(),
            klass=new_class, start=trim_in, end=trim_out, crop=crop or None,
            items_csv=cfg.items_csv)
        return jsonify(to_api(row))
