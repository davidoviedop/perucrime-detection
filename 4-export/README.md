# Etapa 4 — Exportación a RKNN (simulador)

Convierte el modelo final a **Rockchip RKNN-Toolkit2** (v2.3.2) y mide qué se pierde en
el camino. El modelo se parte en dos:

| Parte | Entrada → salida | Dónde correría |
|---|---|---|
| Backbone MobileViT-XS (por fotograma) | imagen `(1,3,H,W)` → embedding `(1,D)` | **NPU** (`.rknn`) |
| Transformer temporal + cabeza | `(1,T,D)` → `(1,n_clases)` | **CPU** (`temporal_head.onnx`) |

La parte temporal solo ve embeddings, no píxeles, por lo que es barata en CPU y evita los
operadores de atención secuencial en la NPU.

> **Estado: código escrito pero todavía NO ejecutado.** Ningún paso de esta etapa se ha
> probado; los resultados y los riesgos de conversión (abajo) están pendientes de verificar.

## Alcance: solo simulador

No hay placa. El simulador de RKNN-Toolkit2 corre en CPU (no hay forma documentada de usar GPU).

| Se puede medir | No se puede medir sin placa |
|---|---|
| Si el backbone convierte, y los avisos de operadores no soportados (log) | Latencia / FPS en la NPU |
| Tamaño del `.rknn` por variante | Qué capas caen en NPU y cuáles en CPU |
| Fidelidad numérica (coseno / L2) del embedding frente a PyTorch | Consumo, memoria en tiempo de ejecución |
| AUC por ventana y por video con el backbone cuantizado | |
| Error por capa (`accuracy_analysis`) | |

**Plataforma destino: `rv1126b`** (NPU de 3 TOPS, una de las más ligeras de la familia; admite
`w8a8` y `w4a16`). Se cambia con `--target-platform`.

## Dos entornos

El toolkit fija `numpy<=1.26.4`, `torch<=2.4.0` y `protobuf<=4.25.4`, incompatibles con el
entorno del proyecto. Por eso:

| Paso | Entorno | Qué hace |
|---|---|---|
| `1_export_onnx.py` | proyecto | ONNX del backbone y de la cabeza, imágenes de calibración, fotogramas de test |
| `2_build_rknn.py` | `~/venvs/rknn` | ONNX → `.rknn` y embeddings del simulador (solo numpy + rknn) |
| `3_score.py` | proyecto | métricas de la tesis con cada variante |

El entorno del toolkit se crea con `bash setup_rknn_env.sh` (descarga el wheel `cp312`, verifica
su md5 y hace una prueba de importación).

## Uso

```bash
# 1. (entorno del proyecto) exportar el estudiante destilado de la etapa 3
# (los pesos van en workspace/weights/<id>/ con best.pt + run.yaml; --data-root solo si el data_root
#  de run.yaml ya no existe, p. ej. ../data/peru-crime5)
python 1_export_onnx.py --run-dir ../weights/<id_estudiante> [--data-root ../data/peru-crime5]

# 2. (entorno del toolkit) primero medir la velocidad del simulador, luego correr completo
source ~/venvs/rknn/bin/activate
python 2_build_rknn.py --export-dir artifacts/<id> --builds w8a8 --probe
python 2_build_rknn.py --export-dir artifacts/<id> --workers 4        # las construcciones de `run:` en builds.yaml

# 3. (entorno del proyecto) puntuar
python 3_score.py --export-dir artifacts/<id>        # -> results/rknn_export.csv
```

`2_build_rknn.py` es reanudable (trozos de `--chunk-size` fotogramas guardados en `emb_cache/<nombre>/`).
Cada carpeta de trozos guarda la firma de su construcción: si cambias algoritmo, calibración, ONNX, etc.,
los trozos viejos se borran y se recalculan en lugar de reutilizarse.

### Construcciones (`builds.yaml`)

Lo que construye `2_build_rknn.py` se define en `builds.yaml`: cada entrada de `builds` tiene un
**nombre**, una `variant` base y sus parámetros (`algorithm`, `quantized_method`, `calib_limit`,
`hybrid_cos_thresh`, `opt_level`, `flash_attention`); lo que falte sale de `defaults`. Todos los archivos
llevan el nombre (`backbone_<nombre>.rknn`, `emb_<nombre>.npy`, `build_info_<nombre>.json`), así que
varias opciones de cuantización conviven y `3_score.py` las puntúa como filas distintas.

```bash
python 2_build_rknn.py --export-dir artifacts/<id> --builds w8a8_kl w8a8_mmse w8a8_hybrid
python 3_score.py --export-dir artifacts/<id>      # puntúa todos los emb_*.npy
```

Para probar otra opción, añade una entrada al YAML (p. ej. `w8a8_mmse_30: {variant: w8a8, algorithm: mmse, calib_limit: 30}`).
`--algorithm`, `--calib-limit`, etc. por CLI pisan el YAML en todas las construcciones elegidas.
`mmse` es muy lento (recomienda 20-50 imágenes de calibración).

### Variantes

| Variante base (`variant`) | Descripción |
|---|---|
| `fp16` | sin cuantizar |
| `w8a8` | int8 en pesos y activaciones, calibrado con ~100 imágenes de **train** |
| `w8a8_hybrid` | int8 con las capas más sensibles en fp16 (`auto_hybrid`) |
| `w4a16` | pesos de 4 bit, activaciones fp16 |

## Prueba de cuantización por backbone (`quant_check.py`)

Aparte del flujo 1→2→3. Para varios backbones de timm **sin entrenar** (pesos de ImageNet) mide cuánto
sobrevive el embedding a la cuantización int8 de RKNN (coseno frente a PyTorch fp32), para decidir a qué
backbone destilar antes de gastar entrenamiento. Corre en el venv del toolkit y reutiliza la calibración y
los fotogramas de test de una exportación del paso 1.

```bash
python quant_check.py --export-dir artifacts/<id>        # resnet18, mobilenetv2/v3, convnextv2_femto, mobilevit_xs (control)
python quant_check.py --export-dir artifacts/<id> --models resnet34 regnety_008 --builds fp16 w8a8 w8a8_kl
```

Salida: `results/quant_check.csv` (una fila por modelo y construcción; un fallo de conversión queda en la columna `error`).
Los pesos de ImageNet no son los afinados, así que es un indicador de familias, no el número final.

## Decisiones de diseño

- **Mismo preprocesado**: el redimensionado es el `T.Resize` de torchvision del entrenamiento; la
  normalización ImageNet va dentro del `.rknn` (`mean/std` en escala 0–255), así que el simulador
  recibe `uint8` RGB.
- **Un embedding por fotograma único**: las ventanas se solapan, así que el test requiere ~3 400
  pasadas por el backbone y no ~10 800.
- **Referencias**: `torch_fp32` y `onnx_fp32` se puntúan con la misma cabeza; si ONNX difiere de
  PyTorch, el problema está en la exportación y no en RKNN.
- **Atención desactivada del camino rápido de PyTorch** (`mha.set_fastpath_enabled(False)`) para
  que el grafo ONNX sea el estándar.

## Riesgos conocidos (sin verificar)

- `Reshape`/`Transpose` de rango 5 en el desdoblado (unfold) de MobileViT: la NPU trabaja con 4-D y
  podría exigir reescritura o mandarlos a CPU. Se verá en `logs/build_<variante>.log`.
- Velocidad del simulador para ~3 400 fotogramas (usar `--probe` antes).
- Comportamiento de `w8a8_hybrid` y `w4a16` en RV1126B.
- Un `.rknn` cargado con `load_rknn` no sirve para el simulador; por eso cada proceso reconstruye el modelo.

## Salidas

`artifacts/{run_id}/` (ignorada por git): ONNX, `.rknn`, calibración, fotogramas, embeddings, logs.
`results/rknn_export.csv`: una fila por variante con tamaño, fidelidad y AUC de ventana/video.

## Reportes de experimentos

Cada vez que se ejecuta un script de esta etapa se escribe un reporte en `reports/` (uno por ejecución, con la fecha en el nombre,
sin sobrescribir los anteriores): `AAAAMMDD-HHMMSS_{script}[_{run_id}].md` para leer y `.json` con lo mismo para procesar. Sirve para
seguir el avance de la etapa y para **reanudar o replicar en otro equipo**: trae el estado (completado / falló / interrumpido), el
comando exacto y desde qué carpeta lanzarlo, todos los parámetros (también los valores por defecto) y la configuración completa
(los YAML se incluyen enteros), la huella de las entradas (tamaño, sha256, archivos por carpeta), el entorno (servidor, Python,
versiones de los paquetes, GPU, ffmpeg y `pip freeze`), los resultados, los archivos de salida y las instrucciones de reanudación.
El reporte se escribe en cuanto se conocen los parámetros, así que si el proceso se cae queda igual. Lo genera `../run_report.py`
(solo biblioteca estándar); no se mezcla con los de las otras etapas.
