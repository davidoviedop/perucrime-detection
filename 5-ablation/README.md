# Etapa 5 — Ablación de modelado

Mide cuánto aporta cada decisión de modelado del detector (ConvNeXtV2-Femto + transformer temporal)
quitando o cambiando **un componente a la vez** (diseño OFAT) y manteniendo todo lo demás fijo. El
diseño completo —modelo base, protocolo, las 14 configuraciones A0–A13, las matrices de puntuación
M1–M12 y el análisis estadístico— está en [`ablation_definition.md`](ablation_definition.md); este
README explica cómo se ejecuta y qué decisiones de implementación se tomaron.

Alcance: solo la etapa 2 (arquitectura, entrenamiento, datos de entrada, puntuación MIL). La
destilación, poda, cuantización y exportación (etapas 3 y 4) se ablacionarán aparte sobre el modelo que
salga de esta suite.

```
 config/pilot_lr.yaml ─► 1_run_ablation.py ─► piloto de LR (A0, 10 épocas, solo validación)
 config/ablation_modelling.yaml ─► 1_run_ablation.py ─► outputs/modelling/{A0..A13}_s{42,43,44}/
        (A0–A13, 3 semillas)            (entrena y evalúa)     best.pt · metrics_val/test.json · preds_val/test.npz
                                                                          │
 0_check_data.py   conteos por clase y resolución de los clips            ├─► 2_report.py            tabla principal, Δ con IC 95 %, veredictos
                                                                          └─► 3_scoring_ablation.py  M1–M12 sin reentrenar
```

## Estructura

```
5-ablation/
├── ablation_definition.md   el diseño de la suite (fuente de verdad)
├── 0_check_data.py          Paso 0  videos y ventanas por clase y split; resolución de los clips (A13)
├── 1_run_ablation.py        Paso 1  entrena y evalúa cada (variante, semilla) → outputs/{suite}/
├── 2_report.py              Paso 2  tablas: media ± de, Δ vs A0 con IC 95 % (bootstrap pareado), veredicto, validación
├── 3_scoring_ablation.py    Paso 3  matrices de puntuación por video M1–M12 (sin entrenar)
├── config/
│   ├── a0_convnextv2_femto.yaml   A0: modelo base y protocolo común
│   ├── ablation_modelling.yaml    la suite: A0–A13, cada una cambia un campo de A0
│   └── pilot_lr.yaml              piloto de LR del backbone (1e-5 vs 3e-5)
├── results/                 tablas generadas (Markdown y CSV)
├── requirements.txt
└── ablation/                lógica propia (no se llama `lib` para no chocar con la etapa 2)
    ├── suite.py             carga la suite, entrena y evalúa cada corrida, guarda métricas, predicciones y variant.json
    ├── scoring.py           M1–M12 sobre las predicciones guardadas
    ├── stats.py             probabilidades promediadas entre semillas, bootstrap pareado estratificado, regla de veredicto
    └── report.py            agrega las corridas en las tablas del informe
```

## Uso

```bash
python 0_check_data.py                                   # conteos por clase y resolución (antes de empezar)
python 1_run_ablation.py --suite config/pilot_lr.yaml    # piloto de LR; fija lr_backbone para toda la suite
python 2_report.py --suite config/pilot_lr.yaml

python 1_run_ablation.py --list                          # las 42 corridas y qué cambia cada una; no corre nada
python 1_run_ablation.py --only A0                       # A0 con sus 3 semillas (mide el tiempo por corrida)
python 3_scoring_ablation.py                             # M1–M12 sobre A0
python 1_run_ablation.py --only A10 --seeds 42 43 44     # solo esas variantes (A0 se incluye siempre)
python 1_run_ablation.py                                 # toda la suite
python 2_report.py                                       # results/ablation_modelling.md
```

* Todos los comandos se ejecutan desde esta carpeta. `--set campo=valor` pisa cualquier campo de
  `Config` (p. ej. `--set grad_accum=2 --set batch_size=8` si una variante no cabe en 16 GB);
  `--set epochs=2` sirve para probar el flujo.
* **Se puede interrumpir y repetir el comando:** una corrida con sus métricas se da por terminada y se
  omite, una con `last.pt` se reanuda, una con `best.pt` pero sin métricas solo se evalúa. Si una
  falla, las demás siguen y al final se listan las que fallaron.
* Las 3 semillas de una configuración deben correr en el mismo servidor. Cada `run.yaml` (sección
  `extra.ablation`) y `variant.json` registran el servidor, la GPU y las versiones de torch, timm y CUDA;
  `variant.json` guarda también `train_seconds`, que responde la pregunta abierta del tiempo por corrida
  de A0 y A10.
* wandb (opcional, `--wandb`): las corridas se llaman `abl-<id>-s<semilla>` (p. ej. `abl-A2-s43`).

## Cómo se implementó el diseño

Casi todo es un campo de `Config`; cada variante de `ablation_modelling.yaml` cambia uno (el listado de
`--list` muestra el cambio). Lo que no existía se agregó a la etapa 2 **sin cambiar el comportamiento
por defecto**:

| Necesidad del diseño | Dónde (etapa 2) | Valor por defecto |
|---|---|---|
| `temporal_module`: `transformer` \| `none` \| `bilstm` (A1, A2) | `lib/config.py`, `lib/model.py` | `transformer` |
| BiLSTM de 1 capa, feat_dim/2 unidades por dirección (192 con dim. 384), promedio temporal, dropout 0.1, sin embedding posicional | `lib/model.py` (`TemporalBiLSTM`) | — |
| Calentamiento lineal por paso (`warmup_epochs`); el recorte de gradiente a 1.0 ya existía | `lib/training.py` | 0 (sin calentamiento) |
| Umbral absoluto de ReduceLROnPlateau (`lr_threshold_mode: abs`) | `lib/training.py` | `rel` |
| Acumulación de gradiente (`grad_accum`) | `lib/training.py` | 1 |
| Probabilidades por ventana de val y test (video, fuente, clase) | `lib/evaluation.py` (`save_predictions`), `2_evaluate.py` | — |
| Agregaciones de M1–M12 como funciones separadas (anomalía, multiclase, decisión dura) | `lib/mil.py` | `mil_aggregate_by_video` no cambió |
| Los LR del optimizador se registran por nombre de grupo (A1 y A6 dejan grupos vacíos) | `lib/training.py` | — |

Congelamiento: el congelamiento parcial de ConvNeXtV2 está escrito explícito (`[stem, stages.0, stages.1]`);
A6 usa el prefijo vacío `[""]` (congela todo el backbone; el optimizador omite el grupo vacío) y A7 `[]`.
A4/A5 cambian `seq_len` y `stride` a la vez (así se mantiene el solapamiento en ≈73 %).

## Decisiones de interpretación

El documento de diseño deja algunos puntos abiertos; esto es lo que se implementó (cámbialo si no es lo que querías):

* **Regla de decisión.** «Δ medio > 1 desviación estándar entre semillas»: se usa |Δ| frente a la **mayor** de las
  desviaciones de A0 y de la variante, y el Δ medio es la diferencia de las medias entre semillas. El IC 95 % sale del
  bootstrap pareado sobre las probabilidades promediadas entre semillas, así que los dos criterios no usan exactamente el
  mismo estimador. Se muestra el signo (↑/↓) junto al veredicto: «aporta» significa que el cambio importa, en cualquier dirección.
* **Bootstrap estratificado por clase.** Se remuestrean videos con reemplazo dentro de cada clase, para que ninguna clase
  desaparezca en un remuestreo (con 4 videos de Robbery o Shooting, un bootstrap simple las pierde ≈1.4 % de las veces).
* **M2 (máximo)** es el top-k con k = 1 ventana. **M1–M5** usan, para F1 y exactitud, la decisión dura actual con ese k (voto
  entre las k ventanas, sin que Normal vote). **M9–M11** usan Normal si el video no supera el umbral (top-k 10 %, 0.50) y, si lo
  supera, el argmax de las clases **no Normal** del vector agregado.
* **Test una sola vez.** Todo lo que se elige (checkpoint, umbral) se elige en validación: el checkpoint por AUC de validación
  por ventana, el umbral de M6–M8 por F1-Macro de validación. El runner evalúa test al terminar cada corrida, pero nada del
  entrenamiento ni de las matrices M lo usa para decidir. El piloto de LR evalúa **solo validación**.
* **GFLOPs** = 2 × MACs, medidos con `FlopCounterMode` sobre una ventana de 16 fotogramas; `FlopCounterMode` no cuenta las
  celdas LSTM (A2), cuyo aporte es despreciable frente al backbone.
* **Split congelado:** la suite lee siempre el mismo `data_root` (`../data/peru-crime5`), cuyos videos ya están asignados a
  train/val/test por carpeta; A4 y A5 cambian el número de ventanas pero no los videos de cada split.

## Observaciones sobre el documento de diseño

* El encabezado de la matriz dice «A0–A12» pero la tabla (y el plan de 42 corridas) tiene A0–A13.
* La pregunta abierta del curriculum learning propone llamarlo «A13», que ya es la resolución 224.
* MobileViT-XS mide 0.36 GMACs por fotograma a 160×160 (el documento dice 0.38); los demás valores de la tabla de backbones
  coinciden con lo medido (parámetros y GMACs de ConvNeXtV2-Atto/Femto/Pico/Nano/Base, timm 1.0, `FlopCounterMode`).
* `python 0_check_data.py` responde dos preguntas abiertas: el test actual tiene **58 videos** (Assault 14, Normal 29,
  Road_Accident 7, Robbery 4, Shooting 4; 678 ventanas) y **6 de los 58 videos de test (todos Normal) miden menos de 224 px**
  de lado menor, así que A13 reescala hacia arriba esos clips: hay que decidir si es aceptable antes de lanzarla.

## Cómo leer los resultados

* El test es pequeño (58 videos; 4 de Robbery y 4 de Shooting): por eso cada Δ lleva IC 95 % y el AUC por clase es solo
  descriptivo.
* Una ablación quita una pieza; no prueba que las demás sean óptimas. Todas las variantes usan los LR de A0 sin reajustarlos.
* El modelo ablacionado es el detector entrenado directamente; el modelo final del trabajo es el estudiante destilado de la etapa 3.

## Cómo se comprobó

* **Cambios en la etapa 2:** un checkpoint existente (`outputs/slfovdc8`) evaluado con el código modificado reproduce las 54
  métricas de test que tenía guardadas, sin diferencia; las funciones separadas de `lib/mil.py` coinciden con
  `mil_aggregate_by_video` (decisión dura, puntaje de anomalía y probabilidades) en 3 360 comparaciones video × configuración.
* **Prueba de extremo a extremo** con los datos reales y una configuración mínima (2 épocas, 3 ventanas por video): A0, A1,
  A2 y A6 con 2 semillas entrenan, se evalúan, y `2_report.py` y `3_scoring_ablation.py` generan sus tablas; el piloto de LR
  (`config/pilot_lr.yaml`) produce solo archivos de validación. Los parámetros coinciden con el documento de diseño (A0 6.04 M,
  A1 4.85 M, A2 5.74 M = 4.85 + 0.89 del BiLSTM) y los GFLOPs por ventana de A0 son 12.7 (0.40 GMACs × 16 × 2). Las cifras de esa
  prueba no significan nada (2 épocas); solo valida el flujo.
* **Calentamiento y acumulación** (prueba unitaria con un modelo de juguete): el LR sube linealmente por paso a lo largo de la época
  de calentamiento y se restaura al terminar; 4 lotes de 8 acumulados dan el mismo resultado que un lote de 32 (diferencia 3e-8); con
  `grad_accum=1` el bucle es idéntico al anterior.
* **No se ha corrido la suite completa:** son 42 entrenamientos de hasta 30 épocas (A10, ConvNeXtV2-Base, dominará el tiempo).
  Las corridas mínimas tardaron ≈2–3 min cada una; `variant.json` guarda el tiempo real por corrida.

## Reportes de experimentos

Cada vez que se ejecuta un script de esta etapa se escribe un reporte en `reports/` (uno por ejecución, con la fecha en el nombre,
sin sobrescribir los anteriores): `AAAAMMDD-HHMMSS_{script}[_{run_id}].md` para leer y `.json` con lo mismo para procesar. Sirve para
seguir el avance de la etapa y para **reanudar o replicar en otro equipo**: trae el estado (completado / falló / interrumpido), el
comando exacto y desde qué carpeta lanzarlo, todos los parámetros (también los valores por defecto) y la configuración completa
(los YAML se incluyen enteros), la huella de las entradas (tamaño, sha256, archivos por carpeta), el entorno (servidor, Python,
versiones de los paquetes, GPU, ffmpeg y `pip freeze`), los resultados, los archivos de salida y las instrucciones de reanudación.
El reporte se escribe en cuanto se conocen los parámetros, así que si el proceso se cae queda igual. Lo genera `../run_report.py`
(solo biblioteca estándar); no se mezcla con los de las otras etapas.
En esta etapa hay un reporte por **corrida** (`..._ablation_run_{id}_s{semilla}.md`, con las métricas de val y test completas, la
suite y la configuración base incluidas, y el comando que repite esa corrida), uno por invocación de `1_run_ablation.py` (qué corridas
se pidieron, ejecutaron, omitieron o fallaron) y uno por cada `0_check_data.py`, `2_report.py` y `3_scoring_ablation.py`.

**Los resultados se conservan entre ejecuciones:** además de `metrics_*.json` y `preds_*.npz` de cada corrida y de su reporte, las
tablas de `results/` (`ablation_{suite}.md/.csv`, `scoring_M1-M12.md`) se copian con fecha a `results/history/` cada vez que se
regeneran, así que una ejecución nueva no pisa la anterior. Repetir una corrida con `--retrain` conserva en `reports/` el reporte de la anterior.
