# Etapa 2 — Modelado

Entrena y evalúa el detector de incidentes sobre el dataset que produce la etapa 1
(`../1-dataset/data/3_dataset`). Es el **modelo profesor**: un backbone de imágenes
(ConvNeXtV2-base) aplicado fotograma a fotograma, un transformer temporal sobre los 16
fotogramas de cada ventana y una cabeza de 5 clases. La etapa 3 lo comprime: lo destila
en un estudiante MobileViT-XS y le aplica poda y cuantización.

```
 3_dataset/{fuente}/{split}/{clase}/*.mp4
        │  ventanas deslizantes (16 fotogramas, 1 de cada 4, a 160×160)
        ▼
 1_train.py ──► outputs/{run_id}/  best.pt · last.pt · run.yaml · history.csv
        │
        ▼
 2_evaluate.py ──► métricas por VENTANA y por VIDEO (puntuación MIL) · metrics_test.json
```

## Estructura

```
2-modelling/
├── 1_train.py        Paso 1  entrena y guarda la corrida en outputs/{run_id}/
├── 2_evaluate.py     Paso 2  evalúa el mejor modelo de una corrida sobre test (o val)
├── config/           teacher_convnextv2_base.yaml: hiperparámetros exactos de la corrida de referencia
├── results/          métricas de referencia (sin pesos) de la corrida wa4jsu91
├── requirements.txt
└── lib/              lógica y utilidades
    ├── config.py     Config (dataclass) + carga/guardado en YAML
    ├── dataset.py    ventanas deslizantes, lectura de fotogramas, DataLoaders
    ├── model.py      FrameBackbone + TemporalTransformer + AnomalyDetector
    ├── training.py   run_epoch, evaluate, fit (checkpoints, scheduler, parada temprana)
    ├── metrics.py    AUC macro / por clase / binario, precisión-recall-F1
    ├── mil.py        puntuación de video en dos etapas (MIL)
    ├── scoring.py    arma el conjunto de métricas de un (split, nivel)
    ├── evaluation.py métricas por ventana + por video de un conjunto de predicciones
    ├── tracking.py   history.csv local y, opcionalmente, Weights & Biases
    ├── runs.py       carpeta de una corrida (ids, run.yaml)
    └── cli.py        opciones comunes (--set, --data-root, --wandb)
```

## El modelo

```
ventana (B, 16, 3, 160, 160)
  └─ backbone timm sin clasificador, por fotograma ──► (B, 16, D)     D = 1024 (ConvNeXtV2-base)
     └─ TemporalTransformer: embedding posicional + 1 capa de atención (4 cabezas) + promedio temporal ──► (B, D)
        └─ cabeza: LayerNorm → Dropout(0.5) → Linear(D, 5) ──► logits
```

* Las primeras etapas del backbone se congelan (`freeze_prefixes`) y el resto se ajusta con
  un LR muy bajo (`1e-6`); el transformer temporal y la cabeza, que parten de cero, usan `1e-5`.
* Aumentación espacial de train (Resize, volteo horizontal p=0.5, ColorJitter brillo 0.3 /
  contraste 0.3 / saturación 0.2 / tono 0.05) **por ventana**: los parámetros aleatorios se
  sortean una vez y se aplican iguales a los 16 fotogramas (`WindowTrainTransform`). Las
  corridas de referencia `wa4jsu91` y `8ac40pob` se entrenaron con la versión anterior, que
  sorteaba por fotograma (volteo y color distintos dentro de una misma ventana). Val/test
  no cambian.
* Aumentación temporal opcional (`temporal_jitter`, solo train): el inicio de cada ventana
  se desplaza al azar entre 0 y `temporal_jitter` fotogramas en cada lectura. Con
  `stride=16` y `frame_stride=4`, sin ella dos ventanas vecinas comparten 12 de 16
  fotogramas idénticos; con `--set temporal_jitter=15` cada época ve fotogramas distintos
  sin cambiar la cantidad de ventanas ni el split. Por defecto es 0 (como las corridas de
  referencia); val y test nunca se desplazan.
* Pérdida: entropía cruzada con *label smoothing* 0.2. No se pondera por clase: el
  muestreador de entrenamiento ya balancea las clases por frecuencia inversa.
* AdamW, AMP (precisión mixta) en CUDA, `ReduceLROnPlateau` sobre `val_auc`, parada temprana.

## Entrenar a nivel de ventana, reportar a nivel de video

El modelo se entrena con **ventanas**: cada una hereda la etiqueta de su video (se acepta
el riesgo de que en un clip largo el incidente ocupe solo una parte). Pero lo que importa
al final es el veredicto por **video**, y ahí un voto por mayoría simple falla: si el
incidente solo ocupa una fracción del clip, el modelo acierta en las ventanas tranquilas
(Normal), el voto lo derrota y el video queda mal puntuado *por haber acertado en lo
aburrido*. Tampoco sirve el *max-pooling* clásico de MIL con 5 clases: una sola ventana
espuria puede secuestrar el máximo de una clase, y el riesgo crece con el largo del clip.

Por eso `lib/mil.py` puntúa cada video en dos etapas:

1. **¿Es anómalo?** Se promedia el `P(no Normal)` de las `k` ventanas más anómalas
   (`k = ⌈10 % de las ventanas⌉`, `mil_top_k_fraction`). Hacen falta *varias* ventanas
   seguras (no un pico), pero solo una *minoría* del clip (no la mayoría). Si el promedio
   supera 0.5 (`mil_anomaly_threshold`), el video es anómalo.
2. **¿Cuál?** Voto por mayoría restringido a esas mismas `k` ventanas, sin dejar votar a
   Normal. El AUC binario del video usa ese puntaje continuo.

Esto solo cambia **cómo se reportan** las métricas de validación y test; el entrenamiento
y la elección del mejor checkpoint (mayor `val_auc` por ventana) no dependen de ello.

## Instalación y uso

```bash
pip install -r requirements.txt          # GPU recomendada (la corrida de referencia, ConvNeXtV2-base, tardó ≈3.9 h completa en una RTX 4070 Ti SUPER)

python 1_train.py                                    # profesor: config/teacher_convnextv2_base.yaml
python 1_train.py --set epochs=40 --wandb            # entrenamiento largo, registrando también en wandb
python 2_evaluate.py --run-id <run_id>               # métricas de test de esa corrida
```

* Todos los comandos se ejecutan desde esta carpeta (`2-modelling/`). Los datos se leen de
  `data_root` (por defecto `../1-dataset/data/3_dataset`); cámbialo con `--data-root`.
* Cualquier campo de la configuración se sobrescribe con `--set campo=valor` (se puede repetir).
* Una corrida es la carpeta `outputs/{run_id}/`. Si ya tiene `best.pt` se da por entrenada
  (`--retrain` fuerza reentrenar); si solo tiene `last.pt`, se **reanuda** desde esa época.
* Sin `--wandb` todo se guarda en local (`history.csv`, `metrics_*.json`); wandb es opcional.
* Para evaluar una corrida hecha en otra máquina basta con copiar su carpeta `outputs/{run_id}/`
  (`best.pt` + `run.yaml`).

## Corrida de referencia (profesor `wa4jsu91`)

La configuración exacta está en `config/teacher_convnextv2_base.yaml` (es la que registró wandb).
Resultados en el split de **test** (58 videos / 678 ventanas), en `results/wa4jsu91/`:

| | AUC macro | AUC binario (Normal vs resto) | Accuracy | F1 macro |
|---|---|---|---|---|
| Ventana | 0.8025 | 0.7450 | 0.457 | 0.367 |
| Video (MIL) | 0.8062 | 0.6825 | 0.431 | 0.308 |

Por clase (recall / AUC), ventana → video:

| Clase | Recall ventana | AUC ventana | Recall video | AUC video |
|---|---|---|---|---|
| Assault | 0.92 | 0.96 | 0.93 | 0.96 |
| Normal | 0.48 | 0.75 | 0.28 | 0.70 |
| Road_Accident | 0.59 | 0.80 | 0.43 | 0.73 |
| Robbery | 0.15 | 0.80 | 0.25 | 0.84 |
| Shooting | 0.06 | 0.71 | 0.00 | 0.81 |

**Cómo leer estas cifras (importante para la revisión):**

* **Fueron 2 épocas, no un entrenamiento completo.** El valor por defecto del script original
  era 40; la corrida registrada usó `epochs: 2`. El AUC de entrenamiento llegó a 0.98 mientras
  el de validación quedó en 0.77, es decir, ya hay brecha de generalización. Un entrenamiento
  largo con parada temprana es el siguiente paso natural (`--set epochs=40`).
* **El test es pequeño:** 4 videos de Robbery y 4 de Shooting. Con tan pocos videos, un solo
  acierto cambia el recall de esas clases en 25 puntos; las cifras por clase de video son
  muy ruidosas. Se reporta una sola corrida y una sola semilla, sin intervalos de confianza.
* **Shooting y Robbery casi no se detectan** (recall de video 0.00 y 0.25) aunque su AUC es
  razonable: el modelo las ordena bien pero no cruza el umbral de decisión. Son las dos
  clases con menos videos de test.
* **Procedencia de los datos:** esas cifras se calcularon sobre la carpeta `peru-crime5`
  original. Si regeneras el dataset con la etapa 1, los clips se recodifican desde los videos
  crudos y las cifras pueden variar ligeramente.

## Qué cambió respecto del script original

Todo el código viene de `2.5-model/models/mil_opt/opt_convbase_v2.py` (el profesor) y de las
utilidades que importa; el comportamiento del modelo, el muestreo, la pérdida, el scheduler y
las métricas se conservaron. Lo que se reorganizó:

| Original | Aquí |
|---|---|
| El script entero con constantes al inicio | `Config` + YAML (`config/`) + `--set`; una corrida guarda su `run.yaml` |
| `utils/datasets/perucrime4.py` | `lib/dataset.py` |
| `utils/mil_training.py` | `lib/metrics.py`, `lib/mil.py`, `lib/training.py` (se descartaron `class_weighted_criterion` y `build_weighted_sampler`: ningún script final las usa) |
| `_scoring_dict` (mezclaba números y gráficos de wandb) | `lib/scoring.py` (solo números; los gráficos van aparte, al `Tracker`) |
| Secciones 8-9 del script (test y barrido de compresión) | `2_evaluate.py` y la etapa 3 |
| `MobileViTBackbone` / `ViTAnomalyDetector` (nombres heredados) | `FrameBackbone` / `AnomalyDetector`: **los nombres de los atributos no cambiaron**, así que los checkpoints existentes cargan tal cual |
| `pip install` dentro del script | `requirements.txt` |
| wandb obligatorio | wandb opcional (`--wandb`); siempre se escribe `history.csv` en local |

Las constantes de ventana (`seq_len`, `stride`, `frame_stride`, `max_windows_per_video`) deben
coincidir con las de `1-dataset/lib/splitting/config.py`, que las usa para balancear la
división por cantidad de ventanas.

## Cómo se comprobó

* **Reproducción de métricas registradas.** Con los checkpoints `best.pt` ya existentes y el
  `run.yaml` armado a partir de la configuración de wandb, `2_evaluate.py` reproduce **las 54
  métricas de test que registró wandb, sin ninguna diferencia**, tanto para el profesor
  (`wa4jsu91`) como para el estudiante de la etapa 3 (`8ac40pob`): AUC macro/binario por
  ventana y por video, AUC/precisión/recall/F1 por clase, accuracy y conteos. Eso valida a la
  vez el cargador de datos, el modelo, las métricas y la puntuación MIL.
* **Entrenamiento** (datos sintéticos): entrena y alcanza AUC 1.0 en un problema separable;
  la corrida se **reanuda** desde `last.pt` (épocas 4-5 tras 3 épocas) y `history.csv`
  conserva las filas anteriores; con `best.pt` presente **se omite** el entrenamiento;
  la **parada temprana** corta la corrida; un checkpoint **incompatible** se descarta y se
  empieza de cero; con `--wandb` (modo offline) se registran métricas, matrices de confusión
  y artefactos sin errores.

* **Corrida real de humo.** Con el dataset real se entrenó 1 época con backbone `mobilevit_xs`
  (misma arquitectura y LR del estudiante de la etapa 3, pero **sin destilación**, solo
  entropía cruzada): carga los pesos preentrenados, recorre las 198 iteraciones de
  entrenamiento y la validación, guarda `best.pt` y `2_evaluate.py` lo evalúa (val AUC 0.840;
  test: AUC por ventana 0.780, por video 0.795). **No es una comparación controlada** (1 época,
  una semilla), pero sugiere que ese estudiante sin destilar ya es competitivo con las cifras
  del estudiante destilado de la etapa 3; conviene medirlo con el mismo número de épocas.

**No se re-entrenó el profesor aquí** (la corrida de referencia tardó ≈3.9 h): la comprobación de
entrenamiento es funcional, no una reproducción bit a bit. Además el entrenamiento en GPU no
es determinista (kernels de cuDNN, orden de los workers), así que dos corridas con la misma
semilla no darán exactamente los mismos números.
