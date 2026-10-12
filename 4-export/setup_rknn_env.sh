#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_rknn_env.sh — instala el entorno de RKNN-Toolkit2 para la exportación del modelo
# (solo SIMULADOR en PC, sin placa).
#
# Qué hace
#   1. Crea un entorno virtual APARTE (~/venvs/rknn). No se usa el del proyecto porque el
#      toolkit fija numpy<=1.26.4, torch<=2.4.0 y protobuf<=4.25.4, que chocan con el
#      numpy 2.x y el torch 2.12 del proyecto.
#   2. Instala torch 2.4.0 solo CPU (evita bajar ~3 GB de CUDA que el toolkit no usa).
#   3. Descarga de GitHub (airockchip/rknn-toolkit2, v2.3.2) el wheel y el requirements
#      para Python 3.12, verifica el md5 del wheel e instala.
#   4. Comprueba que `from rknn.api import RKNN` funciona.
#   5. (Opcional) Instala onnx y onnxruntime en el entorno del proyecto, que es donde se
#      exporta el modelo a ONNX.
#
# Uso
#   bash setup_rknn_env.sh                   # pasos 1-4 y 5
#   SKIP_PROJECT_ENV=1 bash setup_rknn_env.sh   # omite el paso 5
#
# Variables (todas opcionales)
#   RKNN_VENV     dónde crear el entorno del toolkit      (por defecto ~/venvs/rknn)
#   DL_DIR        dónde guardar las descargas              (por defecto ~/rknn-dl)
#   PROJECT_VENV  entorno del proyecto, para onnx          (por defecto /home/chech/tesis/.venv)
#   PYTHON        intérprete a usar                         (por defecto python3.12)
#
# Requisitos: Linux x86_64 (WSL sirve), Python 3.12, curl y conexión a internet.
# Después de instalar:  source ~/venvs/rknn/bin/activate
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RKNN_VENV="${RKNN_VENV:-$HOME/venvs/rknn}"
DL_DIR="${DL_DIR:-$HOME/rknn-dl}"
PROJECT_VENV="${PROJECT_VENV:-/home/chech/tesis/.venv}"
PYTHON="${PYTHON:-python3.12}"

VERSION="2.3.2"
PYTAG="cp312"                      # debe coincidir con la versión de Python de $PYTHON
BASE="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v${VERSION}/rknn-toolkit2/packages/x86_64"
WHEEL="rknn_toolkit2-${VERSION}-${PYTAG}-${PYTAG}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
REQS="requirements_${PYTAG}-${VERSION}.txt"

paso() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ── 0. Comprobaciones ─────────────────────────────────────────────────────────
paso "0/5 Comprobando requisitos"
[ "$(uname -s)" = "Linux" ]   || { echo "ERROR: RKNN-Toolkit2 solo corre en Linux (usa WSL en Windows)."; exit 1; }
[ "$(uname -m)" = "x86_64" ]  || { echo "ERROR: hace falta una máquina x86_64 (el wheel es x86_64)."; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "ERROR: no se encuentra $PYTHON."; exit 1; }
command -v curl >/dev/null      || { echo "ERROR: falta curl."; exit 1; }
PYVER="$("$PYTHON" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
[ "$PYVER" = "$PYTAG" ] || { echo "ERROR: $PYTHON es $PYVER pero el wheel descargado es $PYTAG. Ajusta PYTAG/WHEEL/REQS."; exit 1; }
echo "Linux x86_64, $PYTHON ($PYVER): OK"

# ── 1. Entorno virtual aparte ─────────────────────────────────────────────────
paso "1/5 Creando el entorno virtual en $RKNN_VENV"
if [ -d "$RKNN_VENV" ]; then
  echo "Ya existe $RKNN_VENV: se reutiliza (bórralo a mano si quieres empezar de cero)."
else
  mkdir -p "$(dirname "$RKNN_VENV")"
  "$PYTHON" -m venv "$RKNN_VENV"
fi
# shellcheck disable=SC1091
source "$RKNN_VENV/bin/activate"
python -m pip install --upgrade pip

# ── 2. torch solo CPU ─────────────────────────────────────────────────────────
paso "2/5 Instalando torch 2.4.0 (CPU)"
pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cpu

# ── 3. Descargar e instalar RKNN-Toolkit2 ─────────────────────────────────────
paso "3/5 Descargando RKNN-Toolkit2 v${VERSION} a $DL_DIR"
mkdir -p "$DL_DIR"
cd "$DL_DIR"
curl -fLO "$BASE/$REQS"
curl -fLO "$BASE/$WHEEL"
curl -fLO "$BASE/md5sum.txt"
echo "Verificando el md5 del wheel…"
md5sum -c --ignore-missing md5sum.txt

paso "3/5 Instalando dependencias y el wheel"
pip install -r "$REQS"
pip install "$WHEEL"

# ── 4. Prueba de humo ─────────────────────────────────────────────────────────
paso "4/5 Comprobando la instalación"
pip show rknn-toolkit2 | sed -n '1,2p'
python -c "from rknn.api import RKNN; print('RKNN OK (importa correctamente)')"
deactivate

# ── 5. onnx / onnxruntime en el entorno del proyecto (para exportar a ONNX) ───
if [ "${SKIP_PROJECT_ENV:-0}" = "1" ]; then
  paso "5/5 Omitido (SKIP_PROJECT_ENV=1)"
elif [ -x "$PROJECT_VENV/bin/pip" ]; then
  paso "5/5 Instalando onnx y onnxruntime en $PROJECT_VENV"
  "$PROJECT_VENV/bin/pip" install onnx onnxruntime
  # Solo si tu torch ya no ofrece el exportador clásico (torch.onnx.export(..., dynamo=False)):
  # "$PROJECT_VENV/bin/pip" install onnxscript
else
  paso "5/5 No se encontró $PROJECT_VENV/bin/pip: instala a mano 'onnx onnxruntime' en tu entorno del proyecto."
fi

paso "Listo"
cat <<EOF
Entorno del toolkit : source $RKNN_VENV/bin/activate
Se usa solo el simulador: rknn.init_runtime() SIN el argumento target.
Plataforma elegida  : rknn.config(target_platform='rv1126b')
EOF
