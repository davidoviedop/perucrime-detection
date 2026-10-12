"""
Exportación del dataset final: arma {fuente}/{split}/{clase}/ a partir de lo que
decidieron los pasos anteriores, y escribe manifest.csv.

Entradas
  clips.csv     los clips activos de `scrapped` (clase final tras la validación)
  data/2_clips  sus videos generados
  split.csv     el split de cada item (lo asigna 7_split.py)
  data/external las carpetas sueltas de ucf-crime / vioperu

La carpeta final se puede regenerar cuando se quiera: es un derivado de los CSV.
"""
import csv
import os
import shutil
from pathlib import Path

from lib.common.clips import ClipStore
from lib.splitting.config import LOCAL_SOURCE, SPLIT_NAMES, VIDEO_EXTS
from lib.splitting.pool import active_clip_rows, item_key, scan_pool

MANIFEST_COLUMNS = ["item_id", "clip_id", "source", "class", "label", "split", "filename",
                    "output_path", "start_sec", "end_sec", "source_url"]


class MissingSplitError(RuntimeError):
    """Hay items sin split asignado (falta ejecutar 7_split.py)."""


def plan_export(clips: ClipStore, clips_dir: Path, external_dir: Path, sources, splits: dict) -> list[dict]:
    """Lista de archivos a exportar: [{src, source, item_id, class, split, filename, row|None}, ...]."""
    items = scan_pool(clips, clips_dir, external_dir, sources)
    rows_by_clip = {r["clip_id"]: r for r in active_clip_rows(clips)}

    missing = sorted({iid for iid in items if tuple(iid.split(":", 1)) not in splits})
    if missing:
        raise MissingSplitError(f"{len(missing)} items sin split en split.csv (p. ej. {missing[0]}): "
                                "ejecuta 7_split.py antes de exportar")
    plan = []
    for iid, classes_of_item in items.items():
        source, item_id = iid.split(":", 1)
        for cls, data in classes_of_item.items():
            for src_name, _orig, path in data["files"]:
                plan.append({
                    "src": path, "source": src_name, "item_id": item_id, "class": cls,
                    "split": splits[(source, item_id)], "filename": path.name,
                    "row": rows_by_clip.get(path.stem) if src_name == LOCAL_SOURCE else None,
                })
    plan.sort(key=lambda e: (e["source"], e["split"], e["class"], e["filename"]))
    return plan


def destination(out_root: Path, e: dict) -> Path:
    return out_root / e["source"] / e["split"] / e["class"] / e["filename"]


def manifest_rows(out_root: Path, plan: list[dict]) -> list[dict]:
    rows = []
    for e in plan:
        r = e["row"] or {}
        rows.append({
            "item_id": e["item_id"], "clip_id": r.get("clip_id", ""), "source": e["source"],
            "class": e["class"], "label": "Normal" if e["class"] == "Normal" else "Anomaly",
            "split": e["split"], "filename": e["filename"],
            "output_path": destination(out_root, e).relative_to(out_root).as_posix(),
            "start_sec": r.get("start_sec", ""), "end_sec": r.get("end_sec", ""),
            "source_url": r.get("source_url", ""),
        })
    return rows


def _place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if mode == "copy" and not dst.is_symlink() and dst.stat().st_size == src.stat().st_size:
            return                                   # ya está (misma copia): se omite
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def apply_export(out_root: Path, plan: list[dict], mode: str, dry_run: bool) -> tuple[int, int]:
    """Coloca los archivos y reescribe manifest.csv. Devuelve (colocados, borrados_por_obsoletos)."""
    wanted = {destination(out_root, e) for e in plan}

    # archivos de una exportación anterior que ya no corresponden (según el manifest viejo)
    stale = []
    old = out_root / "manifest.csv"
    if old.is_file():
        with open(old, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = out_root / r["output_path"]
                if p not in wanted and (p.exists() or p.is_symlink()):
                    stale.append(p)

    if dry_run:
        for e in plan[:10]:
            print(f"  [dry-run] {e['src'].name} -> {destination(out_root, e).relative_to(out_root)}")
        if len(plan) > 10:
            print(f"  [dry-run] … y {len(plan) - 10} más")
        return len(plan), len(stale)

    for e in plan:
        _place(e["src"], destination(out_root, e), mode)
    for p in stale:
        p.unlink()

    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / "manifest.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(manifest_rows(out_root, plan))
    os.replace(tmp, old)

    for p in sorted(out_root.rglob("*"), key=lambda p: -len(p.parts)):     # carpetas que quedaron vacías
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    return len(plan), len(stale)


def verify_export(out_root: Path, plan: list[dict], sources) -> bool:
    """Comprueba que estén todos los archivos y que ningún item aparezca en más de un split."""
    print("\n=== Verificación ===")
    missing = [d for d in (destination(out_root, e) for e in plan) if not d.exists()]
    for d in missing:
        print(f"  MISSING: {d}")
    print(f"Existencia de archivos: {len(missing)} faltantes" if missing else "Existencia de archivos: OK, están todos")

    # chequeo de fuga independiente: lee los nombres directo de las carpetas de split en disco
    by_item_split = {}
    for source in sources:
        for split in SPLIT_NAMES:
            split_dir = out_root / source / split
            if not split_dir.is_dir():
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                for f in class_dir.iterdir():
                    if f.suffix.lower() in VIDEO_EXTS:
                        by_item_split.setdefault(item_key(source, f), set()).add(split)
    leaks = {iid: s for iid, s in by_item_split.items() if len(s) > 1}
    print(f"Chequeo de fuga (en disco): {len(leaks)} items en más de un split" if leaks
          else "Chequeo de fuga (en disco): OK, ningún item está en más de un split")
    for iid, s in leaks.items():
        print(f"  LEAK: {iid} en {sorted(s)}")
    return not missing and not leaks

