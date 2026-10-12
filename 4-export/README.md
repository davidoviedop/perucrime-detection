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
python 1_export_onnx.py --run-dir ../3-optimization/outputs/<id_estudiante>

# 2. (entorno del toolkit) primero medir la velocidad del simulador, luego correr completo
source ~/venvs/rknn/bin/activate
python 2_build_rknn.py --export-dir artifacts/<id> --variants w8a8 --probe
python 2_build_rknn.py --export-dir artifacts/<id> --variants fp16 w8a8 --workers 4

# 3. (entorno del proyecto) puntuar
python 3_score.py --export-dir artifacts/<id>        # -> results/rknn_export.csv
```

`2_build_rknn.py` es reanudable (trozos de `--chunk-size` fotogramas guardados en `emb_cache/`).

### Variantes

| Variante | Descripción |
|---|---|
| `fp16` | sin cuantizar |
| `w8a8` | int8 en pesos y activaciones, calibrado con ~100 imágenes de **train** |
| `w8a8_hybrid` | int8 con las capas más sensibles en fp16 (`auto_hybrid`) |
| `w4a16` | pesos de 4 bit, activaciones fp16 |

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
