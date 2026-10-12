"""
2_filter.py — PASO 2: metadatos + etiqueta + filtro de relevancia.

Para cada item 'discovered':
  1. Obtiene metadatos livianos con yt-dlp (sin descargar el video).
  2. Guarda duración, vistas, descripción, fecha de subida y título limpio.
  3. Asigna una de las 13 categorías de UCF-Crime (+ Normal) a partir del
     título y la descripción, con las reglas de palabras clave de
     lib/common/label_rules.py.
  4. Aplica reglas duras (límites de duración, vistas, palabras de rechazo,
     --classes).
  5. Fija el estado:
       'approved'  — hay etiqueta y pasa las reglas  -> entra a la cola de descarga
       'rejected'  — no hay etiqueta o falla alguna regla -> la descarga lo omite

Cookies
-------
Los videos de YouTube con restricción de edad (y el contenido tras login en
otras plataformas) fallan al pedir metadatos sin autenticación
("Sign in to confirm your age") y se rechazarían como fetch_failed. Pasa
cookies.txt por plataforma con --youtube-cookies / --facebook-cookies / ... para
obtener también esos. filter_reason incluye un fragmento del error real de
yt-dlp, de modo que en --stats se distinguen los items con restricción de edad
de las URLs realmente muertas.

Uso
---
    python 2_filter.py [--dry-run] [--source youtube] [--batch-limit 200]
    python 2_filter.py --stats      # distribución de etiquetas y motivos de rechazo
    python 2_filter.py --reset      # re-encola los rechazados -> discovered
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 1-dataset/

from lib.common.logger import get_logger
from lib.scraping.filtering.metadata import fetch_metadata
from lib.scraping.filtering.rules import evaluate
from lib.scraping.settings import (add_classes_arg, add_common_args, add_cookie_args,
                                   parse_classes_or_exit, resolve_cookies)
from lib.scraping.store import ItemStore


def print_stats(db: ItemStore, log) -> None:
    log.info("=== Status ===")
    for status, n in sorted(db.status_summary().items(), key=lambda x: -x[1]):
        log.info(f"  {status:20s}: {n}")
    log.info("=== Labels ===")
    for row in db.label_counts():
        log.info(f"  {str(row['provisional_label']):20s}: {row['n']}")
    log.info("=== Rejection reasons ===")
    for row in db.reject_reason_counts():
        log.info(f"  {str(row['filter_reason']):30s}: {row['n']}")


def main():
    parser = argparse.ArgumentParser(description="Filtro de metadatos + asignación de etiqueta")
    add_common_args(parser)
    parser.add_argument("--source",      default=None,
                        help="Filtrar solo los items de esta plataforma (youtube, facebook, …)")
    parser.add_argument("--country",     default=None,
                        help="Filtrar solo los items de este código de país (PE, BR, …)")
    parser.add_argument("--batch-limit", type=int, default=None,
                        help="Máximo de items por corrida (por defecto: todos los descubiertos)")
    parser.add_argument("--sleep",       type=float, default=0.5,
                        help="Segundos entre consultas de metadatos a yt-dlp (por defecto: 0.5)")
    parser.add_argument("--min-duration", type=float, default=8,
                        help="Rechaza items más cortos que esto, en segundos (por defecto: 8)")
    parser.add_argument("--max-duration", type=float, default=180,
                        help="Rechaza items más largos que esto, en segundos (por defecto: 180)")
    parser.add_argument("--min-views",    type=int, default=0,
                        help="Rechaza items con menos vistas que esto; 0 = sin mínimo (por defecto: 0)")
    add_cookie_args(parser)
    add_classes_arg(parser,
                    "Los items cuya etiqueta cae en otra clase se rechazan "
                    "(motivo class_filtered:<clase>); re-encólalos después con --reset.")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Simula la corrida sin escribir en la BD")
    parser.add_argument("--stats",       action="store_true",
                        help="Muestra la distribución de etiquetas y los motivos de rechazo, y termina")
    parser.add_argument("--reset",       action="store_true",
                        help="Devuelve los rechazados a 'discovered' para volver a filtrarlos")
    args = parser.parse_args()

    target_classes = parse_classes_or_exit(parser, args.classes)

    log, _ = get_logger("filter", str(args.logs_dir))

    db  = ItemStore(args.items_csv)
    cfg = {
        "min_duration_sec": args.min_duration,
        "max_duration_sec": args.max_duration,
        "min_view_count":   args.min_views,
        "target_classes":   target_classes,
    }

    if args.stats:
        print_stats(db, log)
        db.close()
        return

    if args.reset:
        db.requeue_rejected()
        log.info("[Filter] Reset: rejected → discovered.")
        db.close()
        return

    cookies_by_source = resolve_cookies(args, log)

    pending = db.get_by_status("discovered")
    if args.source:
        pending = [i for i in pending if i.get("source", "").lower() == args.source.lower()]
    if args.country:
        pending = [i for i in pending if i.get("country", "").upper() == args.country.upper()]
    if args.batch_limit:
        pending = pending[:args.batch_limit]

    log.info("=" * 60)
    log.info(f"[Filter] {len(pending)} items  |  source={args.source or 'all'}"
             f"  country={args.country or 'all'}  dry_run={args.dry_run}")
    log.info(f"  Classes: {', '.join(sorted(target_classes)) if target_classes else 'all (no filter)'}")
    log.info("=" * 60)

    approved = rejected = fetch_fail = 0
    label_counts: dict[str, int] = {}

    for idx, item in enumerate(pending, 1):
        item_id  = item["id"]
        source   = item.get("source", "?")
        country  = item.get("country", "?")
        title_in = (item.get("raw_title") or "")[:60]

        log.info(f"[{idx}/{len(pending)}] {item_id}  [{source:10s}|{country}]  {title_in}")

        cookies = cookies_by_source.get(source.lower())
        meta, err = fetch_metadata(item.get("url", ""), cookies)

        if meta is None:
            reason = f"fetch_failed:{err[:100]}" if err else "fetch_failed"
            log.info(f"    → fetch failed ({err or '?'}) → rejected")
            if not args.dry_run:
                db.update_item(item_id, {"status": "rejected", "filter_reason": reason})
            fetch_fail += 1
            time.sleep(args.sleep)
            continue

        title = meta.get("raw_title") or title_in
        dur   = meta.get("duration")
        views = meta.get("view_count")
        log.info(f"    title    : {title[:70]}")
        log.info(f"    duration : {dur}s   views: {views}   date: {meta.get('upload_date', '?')}")

        ok, reason, label = evaluate(item, meta, cfg)
        verdict = f"✓ APPROVED [{label}]" if ok else f"✗ REJECTED ({reason})"
        log.info(f"    → {verdict}")

        if not args.dry_run:
            updates: dict = {
                "status":          "approved" if ok else "rejected",
                "filter_reason":   reason,
                "raw_title":       title,
                "raw_description": meta.get("raw_description", ""),
                "channel":         meta.get("channel") or item.get("channel", ""),
                "upload_date":     meta.get("upload_date", ""),
                "duration":        dur,
                "view_count":      views,
            }
            if label:
                updates["provisional_label"] = label
                updates["ucf_category"]      = label
            db.update_item(item_id, updates)

        if ok:
            approved += 1
            label_counts[label or "?"] = label_counts.get(label or "?", 0) + 1   # un item aprobado siempre tiene etiqueta
        else:
            rejected += 1

        time.sleep(args.sleep)

    total = approved + rejected + fetch_fail
    log.info("\n" + "=" * 60)
    log.info(f"[Filter] {'Dry-run ' if args.dry_run else ''}Done — {total} processed")
    if total:
        log.info(f"  Approved  : {approved:5d}  ({100*approved/total:.1f}%)")
    log.info(f"  Rejected  : {rejected:5d}")
    log.info(f"  Fetch fail: {fetch_fail:5d}")
    if label_counts:
        log.info("  Label breakdown:")
        for lbl, n in sorted(label_counts.items(), key=lambda x: -x[1]):
            log.info(f"    {lbl:12s}: {n}")
    if not args.dry_run:
        log.info(f"  DB summary: {db.status_summary()}")
    log.info("=" * 60)
    db.close()


if __name__ == "__main__":
    main()
