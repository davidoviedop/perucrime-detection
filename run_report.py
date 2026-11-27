"""
run_report.py — un reporte por cada experimento (cada ejecución de un script) de cualquier etapa.

Cada etapa guarda los suyos en su propia carpeta `reports/` (no se mezclan entre etapas):

    N-etapa/reports/{AAAAMMDD-HHMMSS}_{script}[_{run_id}].md     lectura humana
    N-etapa/reports/{AAAAMMDD-HHMMSS}_{script}[_{run_id}].json   mismo contenido, para procesar

Los reportes no se sobrescriben (llevan la fecha en el nombre): quedan como el historial de la etapa.

Qué contiene (lo necesario para seguir el avance, reanudar en otro equipo y replicar):
  * estado (en curso / completado / falló / interrumpido), inicio, fin y duración
  * el comando exacto, desde qué carpeta, y las instrucciones para reanudar o replicar
  * todos los parámetros (los argumentos de la línea de comandos, con sus valores por defecto) y la
    configuración completa; los YAML/JSON de configuración se incluyen enteros
  * las entradas con su huella (tamaño, sha256, cantidad de archivos por carpeta)
  * el entorno: servidor, SO, Python, versiones de los paquetes, GPU/driver, ffmpeg y `pip freeze`
  * los resultados y métricas, los archivos de salida y las notas

El reporte se escribe en cuanto se conocen los parámetros (estado «en curso») y se completa al terminar:
si el proceso se cae o se corta, el reporte con los parámetros queda igual.

Uso en un script (solo biblioteca estándar; funciona también en el entorno aparte del toolkit RKNN):

    from run_report import RunReport, current
    with RunReport(Path(__file__).resolve().parent, "1_train"):
        main()                       # los argumentos de argparse se capturan solos
    # y dentro del código, en cualquier punto:
    current().set_config("config", cfg.to_dict())
    current().add_input("data_root", data_root)
    current().add_results({"best_val_auc": 0.81}) ; current().add_output(path)

`current()` no hace nada si no hay un reporte activo, así que el código de la etapa se puede usar sin él.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORKSPACE_DIR = Path(__file__).resolve().parent
_STACK: list = []
_PATCHED = False
_ENV_CACHE: dict = {}

PACKAGES = ["torch", "torchvision", "timm", "numpy", "opencv-python", "opencv-python-headless", "scikit-learn", "scipy",
            "pandas", "PyYAML", "Pillow", "tqdm", "wandb", "flask", "yt-dlp", "onnx", "onnxruntime", "rknn-toolkit2"]
EMBED_SUFFIXES = {".yaml", ".yml", ".json"}
EMBED_MAX_BYTES = 64 * 1024
HASH_MAX_BYTES = 256 * 1024 * 1024
NOT_INPUT_ARGS = {"outputs_dir", "output_dir", "out_dir", "results_dir"}

STAGE_HINTS = {
    "1-dataset": ["El estado del pipeline vive en `data/metadata/*.csv` (items, clips, split) y los videos en `data/1_raw`, "
                  "`data/2_clips` y `data/3_dataset`. Para continuar en otro equipo copia `data/` (como mínimo `metadata/` y "
                  "lo que falte de los pasos anteriores) y lanza el paso siguiente; los pasos son reanudables porque el CSV manda.",
                  "Instala `requirements.txt` y, según el paso, `ffmpeg` (ver «Entorno»)."],
    "2-modelling": ["Una corrida es la carpeta `outputs/{run_id}/`. Con `last.pt` se **reanuda** desde esa época si se repite el "
                    "comando con el mismo `--run-id`; con `best.pt` se da por entrenada (`--retrain` la repite).",
                    "En otro equipo: instala `requirements.txt` (mismas versiones que en «Entorno»), deja el dataset en la ruta "
                    "de `data_root` (o pásala con `--data-root`; el sha256 del `manifest.csv` debe coincidir) y copia "
                    "`outputs/{run_id}/` (al menos `run.yaml`, `last.pt`/`best.pt`, `history.csv`)."],
    "3-optimization": ["Necesita el profesor de la etapa 2 (`outputs/{id}/` con `best.pt` y `run.yaml`; ver «Entradas»). "
                       "Una corrida es `outputs/{run_id}/` y se reanuda con el mismo `--run-id`.",
                       "`2_compress.py` agrega filas a `results/comparison.csv`; copia ese archivo si quieres conservar la tabla."],
    "4-export": ["Los pasos se encadenan por `export_meta.json` en la carpeta de exportación (`--export-dir`); copia esa carpeta "
                 "entera al otro equipo. `2_build_rknn.py` corre en el entorno aparte de RKNN-Toolkit2 (ver README de la etapa)."],
    "5-ablation": ["Cada (variante, semilla) es `outputs/{suite}/{id}_s{semilla}/`. Repetir el comando de `1_run_ablation.py` "
                   "omite las corridas terminadas y reanuda las interrumpidas; para continuar en otro equipo copia `outputs/{suite}/` "
                   "(cada corrida con su `run.yaml`) y mantén las 3 semillas de una configuración en el mismo servidor.",
                   "Los resultados se conservan: cada corrida deja `metrics_*.json`/`preds_*.npz` y su propio reporte, y las tablas "
                   "de `results/` se archivan con fecha en `results/history/`."],
}


# ── Utilidades ────────────────────────────────────────────────────────────────

def _now():
    return dt.datetime.now().astimezone()


def _sha256(path, limit=HASH_MAX_BYTES):
    p = Path(path)
    if p.stat().st_size > limit:
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path, max_files=100000):
    """Huella de un archivo (tamaño, sha256) o de una carpeta (archivos por extensión, bytes, hash de la estructura
    y conteos de archivos por subcarpeta hasta 3 niveles)."""
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    if p.is_file():
        st = p.stat()
        return {"type": "file", "path": str(p.resolve()), "bytes": st.st_size, "sha256": _sha256(p),
                "mtime": dt.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}
    n, total, ext, counts = 0, 0, {}, {}
    h = hashlib.sha256()
    truncated = False
    for root, dirs, files in os.walk(p):
        dirs.sort()
        for name in sorted(files):
            fp = Path(root) / name
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            n += 1
            total += size
            ext[fp.suffix.lower() or "(sin ext.)"] = ext.get(fp.suffix.lower() or "(sin ext.)", 0) + 1
            rel = fp.relative_to(p)
            h.update(f"{rel.as_posix()}:{size}\n".encode())
            if len(rel.parts) > 1:
                key = "/".join(rel.parts[:min(len(rel.parts) - 1, 3)])
                counts[key] = counts.get(key, 0) + 1
            if n >= max_files:
                truncated = True
                break
        if truncated:
            break
    out = {"type": "dir", "path": str(p.resolve()), "files": n, "bytes": total, "structure_sha256": h.hexdigest(),
           "extensions": dict(sorted(ext.items(), key=lambda kv: -kv[1])[:8]),
           "files_by_subfolder": dict(sorted(counts.items())[:80])}
    if truncated:
        out["truncated_at"] = max_files
    for key_file in ("manifest.csv", "run.yaml", "export_meta.json"):
        if (p / key_file).is_file():
            out.setdefault("key_files", {})[key_file] = _sha256(p / key_file)
    return out


def _snapshot(path):
    """Estado de un archivo o carpeta para comparar antes/después (filas de un CSV, bytes, archivos)."""
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    if p.is_file():
        out = {"exists": True, "bytes": p.stat().st_size}
        if p.suffix.lower() == ".csv":
            with p.open("rb") as f:
                out["rows"] = max(sum(1 for _ in f) - 1, 0)
        out["sha256"] = _sha256(p, 64 * 1024 * 1024)
        return out
    n = total = 0
    for root, _d, files in os.walk(p):
        for name in files:
            n += 1
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return {"exists": True, "files": n, "bytes": total}


def environment():
    if _ENV_CACHE:
        return _ENV_CACHE
    env = {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
           "python_executable": sys.executable, "cpu_count": os.cpu_count(), "packages": {}}
    for pkg in PACKAGES:
        try:
            env["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            env["torch_cuda"] = str(torch.version.cuda)
            env["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                env["gpu"] = {"name": props.name, "memory_gb": round(props.total_memory / 1e9, 1), "count": torch.cuda.device_count()}
        except Exception:
            pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                env["nvidia_smi"] = out.splitlines()[0]
        except Exception:
            pass
    if shutil.which("ffmpeg"):
        try:
            env["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
        except Exception:
            pass
    try:
        env["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                                           timeout=30).stdout.strip().splitlines()
    except Exception:
        pass
    _ENV_CACHE.update(env)
    return env


def _jsonable(o) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


# ── Captura automática de los argumentos de argparse ─────────────────────────

def _install_argparse_hook():
    global _PATCHED
    if _PATCHED:
        return
    original = argparse.ArgumentParser.parse_args

    def parse_args(self, *a, **k):
        ns = original(self, *a, **k)
        if _STACK:
            _STACK[-1].capture_args(vars(ns))
        return ns

    argparse.ArgumentParser.parse_args = parse_args
    _PATCHED = True


class _Null:
    """Reporte inactivo: acepta cualquier llamada sin hacer nada."""
    def __getattr__(self, name):
        return lambda *a, **k: None


_NULL = _Null()


def current():
    """El reporte activo (el más interno) o uno inactivo que ignora las llamadas."""
    return _STACK[-1] if _STACK else _NULL


# ── El reporte ────────────────────────────────────────────────────────────────

class RunReport:
    def __init__(self, stage_dir, script, run_id=None, watch=(), stopped_ok=False, command=None, title=None):
        """stage_dir: carpeta de la etapa (el reporte va en stage_dir/reports/). script: nombre del script o experimento.
        watch: rutas (archivos o carpetas) cuyo estado se compara antes y después (p. ej. los CSV de metadatos).
        stopped_ok: el script es un servidor que se corta con Ctrl+C; eso es «detenido», no «interrumpido».
        command: comando exacto para reproducir, si se arma a mano (reportes de corridas internas)."""
        self.stage_dir = Path(stage_dir).resolve()
        self.script = script
        self.run_id = run_id
        self.title = title
        self.watch = [Path(w) for w in watch]
        self.stopped_ok = stopped_ok
        self.command_override = command
        self.args, self.configs, self.files, self.inputs, self.outputs = {}, {}, {}, {}, []
        self.info, self.results, self.notes, self.hints = {}, {}, [], []
        self.status, self.error = "en curso", None
        self.started = self.ended = None
        self._t0 = None
        self._paths = set()
        self._before = self._after = None

    # ---- contexto
    def __enter__(self):
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":      # proceso hijo del recargador de Flask: el padre ya tiene reporte
            self._disabled = True
            return self
        _install_argparse_hook()
        _STACK.append(self)
        self.started, self._t0 = _now(), time.time()
        self._before = {str(w): _snapshot(w) for w in self.watch}
        self.argv = list(sys.argv)
        self.cwd = Path.cwd()
        self.write()
        return self

    def __exit__(self, exc_type, exc, tb):
        if getattr(self, "_disabled", False):
            return False
        self.ended = _now()
        if exc_type is None:
            self.status = "completado"
        elif exc_type is KeyboardInterrupt:
            self.status = "detenido por el usuario" if self.stopped_ok else "interrumpido"
        elif exc_type is SystemExit and (exc.code in (None, 0)):
            self.status = "completado"
        else:
            self.status = "falló"
            self.error = ("".join(traceback.format_exception(exc_type, exc, tb)) if exc_type is not SystemExit
                          else f"SystemExit: {exc.code}")
        self._after = {str(w): _snapshot(w) for w in self.watch}
        if _STACK and _STACK[-1] is self:
            _STACK.pop()
        try:
            self.write()
        except Exception as e:                      # el reporte nunca debe tumbar el experimento
            print(f"[run_report] no se pudo escribir el reporte: {e}", file=sys.stderr)
        if self.path_md:
            print(f"\nReporte del experimento: {self.path_md}")
        return False

    # ---- API para el código de la etapa
    def set_run_id(self, run_id):
        self.run_id = str(run_id)

    def capture_args(self, ns):
        for k, v in ns.items():
            self.args[k] = _jsonable(v)
            if isinstance(v, (str, Path)) and str(v):
                p = self._find(v)
                if p is None:
                    continue
                if p.is_file() and p.suffix.lower() in EMBED_SUFFIXES:
                    self.add_file(k, p)
                elif p.exists() and k not in NOT_INPUT_ARGS and (k.endswith("_root") or k.endswith("_dir") or k == "run_dir"):
                    self.add_input(k, p)
        self.write()

    def _find(self, value):
        p = Path(str(value))
        for base in (None, self.stage_dir):
            cand = p if base is None else base / p
            try:
                if cand.exists():
                    return cand
            except OSError:
                pass
        return None

    def set_config(self, name, cfg):
        self.configs[name] = _jsonable(cfg)

    def add_file(self, name, path):
        p = Path(path)
        info = fingerprint(p)
        if p.is_file() and p.stat().st_size <= EMBED_MAX_BYTES:
            try:
                info["text"] = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass
        self.files[name] = info

    def add_input(self, name, path):
        self.inputs[name] = fingerprint(path)

    def add_output(self, path, note=None):
        p = str(Path(path).resolve())
        if not any(o["path"] == p for o in self.outputs):
            self.outputs.append({"path": p, **({"note": note} if note else {})})

    def add_info(self, mapping):
        self.info.update(dict(_jsonable(mapping)))

    def add_results(self, mapping, section=None):
        data = dict(_jsonable(mapping))
        if section:
            self.results.setdefault(section, {}).update(data)
        else:
            self.results.update(data)

    def note(self, text):
        self.notes.append(str(text))

    def add_hint(self, text):
        self.hints.append(str(text))

    def flush(self):
        try:
            self.write()
        except Exception as e:
            print(f"[run_report] no se pudo escribir el reporte: {e}", file=sys.stderr)

    # ---- salida
    @property
    def reports_dir(self):
        return self.stage_dir / "reports"

    def _stem(self):
        ts = (self.started or _now()).strftime("%Y%m%d-%H%M%S")
        return f"{ts}_{self.script}" + (f"_{self.run_id}" if self.run_id else "")

    @property
    def path_md(self):
        return self.reports_dir / (self._stem() + ".md") if self.started else None

    def command(self):
        if self.command_override:
            return self.command_override
        argv = list(self.argv)
        argv[0] = Path(argv[0]).name
        if self.run_id and "run_id" in self.args and "--run-id" not in argv:
            argv += ["--run-id", self.run_id]
        return "python " + shlex.join(argv)

    def to_dict(self):
        started = self.started or _now()
        try:
            rel_stage = self.stage_dir.relative_to(WORKSPACE_DIR).as_posix()
        except ValueError:
            rel_stage = str(self.stage_dir)
        return {
            "stage": self.stage_dir.name, "script": self.script, "run_id": self.run_id, "status": self.status,
            "started": started.isoformat(timespec="seconds"), "ended": self.ended.isoformat(timespec="seconds") if self.ended else None,
            "duration_seconds": round((self.ended - started).total_seconds() if self.ended else time.time() - (self._t0 or time.time()), 1),
            "command": {"run_from": rel_stage, "cwd": str(self.cwd), "command": self.command(), "argv": self.argv},
            "how_to_resume": STAGE_HINTS.get(self.stage_dir.name, []) + self.hints,
            "args": self.args, "configs": self.configs, "config_files": self.files, "inputs": self.inputs, "info": self.info,
            "environment": environment(), "results": self.results, "outputs": self.outputs, "notes": self.notes,
            "state_before": self._before, "state_after": self._after, "error": self.error,
        }

    def write(self):
        if not self.started:
            return
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        new = {self.reports_dir / (self._stem() + ".json"), self.reports_dir / (self._stem() + ".md")}
        for p, content in ((self.reports_dir / (self._stem() + ".json"), json.dumps(d, indent=2, ensure_ascii=False, default=str)),
                           (self.reports_dir / (self._stem() + ".md"), render_markdown(d))):
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, p)
        for old in self._paths - new:               # el run_id se conoció tarde: se renombra el reporte
            try:
                old.unlink()
            except OSError:
                pass
        self._paths = new


# ── Markdown ──────────────────────────────────────────────────────────────────

def _cell(v, limit=200):
    s = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else str(v)
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out


def _fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def render_markdown(d):
    env = d["environment"]
    title = f"{d['stage']} / {d['script']}" + (f" — {d['run_id']}" if d["run_id"] else "")
    L = [f"# Reporte de experimento: {title}", "",
         "| | |", "|---|---|", f"| Estado | **{d['status']}** |", f"| Inicio | {d['started']} |",
         f"| Fin | {d['ended'] or '—'} |", f"| Duración | {d['duration_seconds'] / 60:.1f} min ({d['duration_seconds']:.0f} s) |",
         f"| Servidor | {env.get('host')} · {env.get('platform')} · Python {env.get('python')} |"]
    gpu = env.get("gpu")
    if gpu or env.get("nvidia_smi"):
        L.append(f"| GPU | {gpu['name'] + ' · ' + str(gpu['memory_gb']) + ' GB' if gpu else env['nvidia_smi']} |")
    L += ["", "## Cómo reproducir o reanudar", "", "Desde la carpeta `workspace/" + d["command"]["run_from"] + "/`:", "",
          "```bash", d["command"]["command"], "```", ""]
    L += [f"* {h}" for h in d["how_to_resume"]]
    if d["error"]:
        L += ["", "## Error", "", "```", d["error"].rstrip(), "```"]
    if d["args"]:
        L += ["", "## Parámetros (argumentos de la línea de comandos, con sus valores por defecto)", ""]
        L += _table([[k, v] for k, v in d["args"].items()], ["Parámetro", "Valor"])
    if d["info"]:
        L += ["", "## Datos de la ejecución", ""] + _table([[k, v] for k, v in d["info"].items()], ["Campo", "Valor"])
    for name, cfg in d["configs"].items():
        L += ["", f"## Configuración completa: {name}", "", "```json", json.dumps(cfg, indent=2, ensure_ascii=False), "```"]
    for name, f in d["config_files"].items():
        L += ["", f"## Archivo de configuración `{name}`: `{f.get('path')}`", "", f"sha256 `{f.get('sha256')}`"]
        if f.get("text"):
            L += ["", "```" + ("yaml" if str(f["path"]).endswith((".yaml", ".yml")) else "json"), f["text"].rstrip(), "```"]
    if d["inputs"]:
        L += ["", "## Entradas (huella para comprobar que son las mismas en otro equipo)", ""]
        rows = []
        for name, f in d["inputs"].items():
            if f.get("exists") is False:
                rows.append([name, f["path"], "no existe", "—"])
            elif f["type"] == "file":
                rows.append([name, f["path"], _fmt_size(f["bytes"]), f"sha256 {f['sha256']}"])
            else:
                keys = "; ".join(f"{k} {v}" for k, v in (f.get("key_files") or {}).items())
                rows.append([name, f["path"], f"{f['files']} archivos · {_fmt_size(f['bytes'])}",
                             f"estructura {f['structure_sha256'][:16]}… {keys}".strip()])
        L += _table(rows, ["Entrada", "Ruta", "Tamaño", "Huella"])
        for name, f in d["inputs"].items():
            if f.get("type") == "dir" and f.get("files_by_subfolder"):
                L += ["", f"Archivos por subcarpeta de `{name}`: " + ", ".join(f"`{k}` {v}" for k, v in list(f["files_by_subfolder"].items())[:40])]
    if d["state_before"]:
        L += ["", "## Estado de los archivos vigilados (antes → después)", ""]
        rows = []
        for k, b in d["state_before"].items():
            a = (d["state_after"] or {}).get(k, {})
            def brief(s):
                return "no existe" if not s or s.get("exists") is False else ", ".join(
                    f"{n} {s[n]}" for n in ("rows", "files", "bytes") if n in s)
            rows.append([k, brief(b), brief(a)])
        L += _table(rows, ["Ruta", "Antes", "Después"])
    L += ["", "## Entorno", "", f"* Python {env.get('python')} (`{env.get('python_executable')}`)"]
    if env.get("packages"):
        L.append("* Paquetes: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()))
    if env.get("torch_cuda"):
        L.append(f"* CUDA de torch {env['torch_cuda']} · disponible: {env.get('cuda_available')}")
    if env.get("nvidia_smi"):
        L.append(f"* nvidia-smi: {env['nvidia_smi']}")
    if env.get("ffmpeg"):
        L.append(f"* {env['ffmpeg']}")
    L.append("* La lista completa (`pip freeze`) está en el `.json` de este reporte.")
    if d["results"]:
        L += ["", "## Resultados", ""]
        flat = [[k, v] for k, v in d["results"].items() if not isinstance(v, dict)]
        if flat:
            L += _table(flat, ["Resultado", "Valor"])
        for k, v in d["results"].items():
            if isinstance(v, dict):
                items = list(v.items())
                L += ["", f"### {k}" + (f" (primeras 60 de {len(items)}; todas en el `.json`)" if len(items) > 60 else ""), ""]
                L += _table([[a, b if not isinstance(b, float) else f"{b:.6g}"] for a, b in items[:60]], ["Métrica", "Valor"])
    if d["outputs"]:
        L += ["", "## Archivos de salida", ""] + [f"* `{o['path']}`" + (f" — {o['note']}" if o.get("note") else "") for o in d["outputs"]]
    if d["notes"]:
        L += ["", "## Notas", ""] + [f"* {n}" for n in d["notes"]]
    return "\n".join(L) + "\n"
