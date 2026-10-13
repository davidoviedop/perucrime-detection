# Etapa 3 — Optimización

Comprime el modelo de la etapa 2 para poder desplegarlo en una cámara: **destilación**
(un estudiante pequeño aprende del profesor), **poda** y **cuantización**, y mide qué se
pierde y qué se gana en cada variante. El resultado es una tabla de comparación
(`results/comparison.csv`) con calidad y costo lado a lado.

```
 profesor (etapa 2)                    estudiante                    variantes comprimidas
 ConvNeXtV2-base                       MobileViT-XS                  poda 0/10/20 % × fp32/fp16/int8
 96.1 M parám · 250.7 GFLOPs   ──►     3.1 M parám · 12.1 GFLOPs ──► calidad · tamaño · GFLOPs · FPS
        1_distill.py (destilar)                2_compress.py (podar · cuantizar · medir)
```

## Estructura

```
3-optimization/
├── 1_distill.py     Paso 1  destila el profesor en un estudiante → outputs/{run_id}/
├── 2_compress.py    Paso 2  poda × cuantización de una corrida + mediciones → results/comparison.csv
├── config/          student_mobilevit_xs_distill.yaml: hiperparámetros exactos del estudiante de referencia
├── results/         comparison.csv y las métricas de referencia (sin pesos) del estudiante 8ac40pob
├── requirements.txt
└── optimization/    lógica propia de la etapa
    ├── config.py        DistillConfig = Config de la etapa 2 + parámetros del profesor y de la pérdida
    ├── distill.py       KDTrainWrapper, DistillationLoss, carga del profesor
    ├── compress.py      prune_model, cast_dtype, predict
    ├── benchmark.py     tamaño, GFLOPs y FPS (GPU y CPU)
    └── comparison.py    comparison.csv y detalle por variante
```

> Esta etapa **reutiliza el código de la etapa 2** (`../2-modelling/lib`: modelo, datos,
> bucle de entrenamiento, métricas, MIL). Por eso su lógica propia no se llama `lib`: sería
> ambiguo. Al importar `optimization` se agrega `2-modelling/` al `sys.path`.

## 1. Destilación (`1_distill.py`)

El estudiante (MobileViT-XS + el mismo transformer temporal y cabeza) se entrena con

```
loss = α · CE(estudiante, y)  +  (1 − α) · T² · KL( softmax(profesor / T) ‖ softmax(estudiante / T) )
```

con `T = 4` (suaviza las dos distribuciones para que se vean las probabilidades pequeñas de
las clases equivocadas) y `α = 0.3` (el 30 % de la pérdida viene de las etiquetas, el 70 %
del profesor); el factor `T²` mantiene comparable la escala del gradiente. Solo se destilan
los **logits** de salida (no hay alineación de características intermedias).

* **El profesor es el modelo completo**, sin poda ni cuantización: se carga desde su corrida
  de la etapa 2, se congela y se deja en modo `eval`. La compresión se aplica después, solo al
  estudiante.
* **El profesor corre dentro del forward del estudiante.** `run_epoch` solo entrega a la
  pérdida `(logits, etiquetas)`, no el lote de entrada, así que la pérdida no puede llamar al
  profesor. `KDTrainWrapper` lo ejecuta durante el entrenamiento y guarda sus logits;
  `DistillationLoss` los recoge. El profesor no es un submódulo registrado, así que el
  optimizador y el `state_dict` son del estudiante.
* En validación y test se usa el estudiante solo y la pérdida es entropía cruzada simple, por
  lo que `val_loss` **no es comparable** con la pérdida de entrenamiento (que incluye el término KL).
* El entrenamiento, los checkpoints, la reanudación y la validación con puntuación MIL son los
  de la etapa 2 (`lib/pipeline.py`); aquí solo cambian el modelo, la pérdida y el profesor.

Una corrida destilada es una carpeta `outputs/{run_id}/` igual a la de la etapa 2 (más la
sección `extra.distillation` en `run.yaml`), así que se evalúa con
`python ../2-modelling/2_evaluate.py --run-id <id> --outputs-dir ../3-optimization/outputs`.

## 2. Poda y cuantización (`2_compress.py`)

Para cada combinación (poda × tipo de dato) se hace una **copia** del modelo (el original y su
`best.pt` no se tocan) y se evalúa en test:

| Técnica | Qué hace | Qué NO hace |
|---|---|---|
| **Poda** global no estructurada por magnitud L1 (0, 10, 20 %) sobre todos los `Conv2d`/`Linear` | Pone en cero los pesos más pequeños, sin ajuste fino posterior | **No reduce tamaño ni GFLOPs ni, en general, tiempo**: los ceros siguen almacenados y multiplicados (no hay formato disperso). Solo muestra cuánta precisión sobrevive |
| **fp16** | Todo el modelo en media precisión, en GPU | En CPU no hay kernels útiles, por eso su `FPS_cpu` es `nan` |
| **int8** | Cuantización dinámica de backbone y cabeza, en CPU | El transformer temporal queda en fp32: su ruta rápida lee `linear1.weight` como tensor y la cuantización dinámica lo convierte en un método. Los convs no se cuantizan (solo `nn.Linear`) |

La poda se aplica **antes** de convertir el tipo de dato (necesita la resolución fp32 para
elegir pesos, y los módulos int8 ya no son `Linear`/`Conv2d`). Con poda sin ajuste fino,
valores muy por encima de ~0.2-0.3 cuestan precisión real.

### Qué se mide

| Columna | Definición |
|---|---|
| `test_window_auc_*`, `test_video_auc_*` | AUC macro (uno-contra-resto) y binario (Normal vs resto) en test, por ventana y por video (puntuación MIL de la etapa 2) |
| `num_parameters` | Parámetros de la arquitectura (igual en todas las variantes: la poda pone ceros e int8 empaqueta sus pesos fuera de `.parameters()`) |
| `size_mb` | Tamaño del `state_dict` serializado de cada variante |
| `GFLOPS` | GFLOPs de **una ventana** de 16 fotogramas (lote 1) contados con `torch.utils.flop_counter` sobre el modelo denso. Es por ventana, no por segundo, y es igual en todas las variantes |
| `FPS` | Fotogramas de entrada por segundo, lote 1 (una ventana por pasada), en `FPS_device`. Mediana de 50 pasadas sincronizadas tras 20 de calentamiento |
| `FPS_device` | Dispositivo donde se midió `FPS` (`cuda`, o `cpu` para int8) |
| `FPS_cpu` | El mismo FPS forzado a CPU para **comparar todas las variantes en igualdad de condiciones** (menos fp16: `nan`). Pocas iteraciones (2 + 5) porque la CPU es lenta |

## Uso

```bash
pip install -r requirements.txt

# 1) destilar (necesita un profesor entrenado en la etapa 2)
python 1_distill.py --teacher-run-dir ../2-modelling/outputs/<id_profesor>
python 1_distill.py --set epochs=40 --wandb                 # entrenamiento largo

# 2) comprimir y medir (profesor o estudiante)
python 2_compress.py --run-dir outputs/<id_estudiante>
python 2_compress.py --run-dir ../2-modelling/outputs/<id_profesor> --skip-cpu-fps
python 2_compress.py --run-dir outputs/<id> --prune-amounts 0 0.2 --dtypes fp32 int8 --verbose
```

Todos los comandos se ejecutan desde esta carpeta. `results/comparison.csv` se comparte entre
corridas: repetir una corrida reemplaza solo sus filas (mismo `run_id`).
