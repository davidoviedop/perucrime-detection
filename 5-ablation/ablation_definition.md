# Diseño de la suite de ablación — Modelado

Nov 22, 2026 · @Cesar

## Objetivo y alcance

La suite mide cuánto aporta cada decisión de modelado del detector ConvNeXtV2-Femto + transformer temporal, quitando o cambiando un componente a la vez y manteniendo todo lo demás fijo. Son 14 configuraciones de entrenamiento (A0–A13, 3 semillas cada una) y 12 evaluaciones de puntuación por video sin reentrenar (M1–M12).

- **Dentro:** etapa 2 (`2-modelling`): arquitectura, entrenamiento, datos de entrada y puntuación MIL por video.
- **Fuera:** etapa 3 y 4 (destilación, poda, cuantización, exportación RKNN). Se ablacionarán aparte, sobre el modelo que salga de esta suite.
- **Diseño:** un factor a la vez (OFAT) respecto de un único modelo base A0. No se buscan interacciones entre factores.
- **Estado:** solo diseño; nada de esto está implementado todavía (ver la última sección).

## Modelo base A0 y protocolo común

A0 es ConvNeXtV2-Femto + transformer temporal. Femto es el tamaño de ConvNeXtV2 más cercano a MobileViT-XS en cómputo (0.40 vs. 0.38 GMACs por fotograma) y comparte su dimensión de características (384), así que el módulo temporal y la cabeza son idénticos entre ambos.

| Backbone | Parámetros (M) | GMACs por fotograma | Dim. de características |
| --- | --- | --- | --- |
| MobileViT-XS (modelo M-2, A3) | 1.93 | 0.38 | 384 |
| ConvNeXtV2-Atto | 3.39 | 0.28 | 320 |
| **ConvNeXtV2-Femto (A0)** | **4.85** | **0.40** | **384** |
| ConvNeXtV2-Pico | 8.55 | 0.70 | 512 |
| ConvNeXtV2-Nano | 14.98 | 1.25 | 640 |
| ConvNeXtV2-Base (profesor, A10) | 87.69 | 7.83 | 1024 |

Medido con timm 1.0 y `FlopCounterMode` de PyTorch a 160×160, sin cabeza de clasificación. Atto tiene menos cómputo pero cambia la dimensión a 320; Pico y Nano duplican o triplican el cómputo. El análogo en MobileViTv2 sería `mobilevitv2_075` (2.48 M, 0.41 GMACs, 384), también alineado con Femto.

| Aspecto | Valor en A0 |
| --- | --- |
| Backbone | `convnextv2_femto` (ImageNet); congelamiento parcial: `stem`, `stages.0`, `stages.1` |
| Módulo temporal | TemporalTransformer: 1 capa, 4 cabezas, embedding posicional, promedio temporal (1.18 M parámetros) |
| Cabeza | LayerNorm → Dropout 0.5 → Linear(384, 5) |
| Ventana | `seq_len` 16, `stride` 16, `frame_stride` 4 (61 fotogramas fuente, ≈2.4 s), 160×160, máx. 70 ventanas/video |
| Pérdida y balanceo | Entropía cruzada, `label_smoothing` 0.2; `WeightedRandomSampler` fijo en toda la suite |
| Aumento | flip + ColorJitter por ventana, `temporal_jitter` 8 |
| Optimizador | AdamW, weight decay 1e-3; LR 3e-5 / 3e-4 / 3e-4 (backbone / temporal / cabeza); warmup lineal de 1 época; recorte de gradiente a norma 1.0 |
| Scheduler | `ReduceLROnPlateau` sobre AUC de validación: factor 0.5, patience 1, umbral absoluto 1e-3, LR mínimo 1e-7 |
| Parada | Máx. 30 épocas, early stopping con patience 5 |
| Batch | 16 efectivo; acumulación de gradiente si una variante no cabe en 16 GB |
| Selección | Mejor checkpoint por AUC-Macro de validación por ventana |

**Por qué estos LR.** El protocolo anterior (1e-6 / 1e-5 / 1e-5) casi no movía el backbone. El nuevo tope del backbone, 3e-5, queda por debajo del rango que ya destruyó pesos preentrenados (5e-4); 3e-4 para los módulos que parten de cero es el valor ya usado en la destilación del estudiante. El scheduler baja el LR tras 1 época sin mejorar al menos 0.001 de AUC y corta tras 5.

**Piloto de LR (no es parte de la ablación).** Antes de la matriz: A0, semilla 42, 10 épocas, backbone 1e-5 vs. 3e-5, comparado solo en validación. Si 3e-5 queda por debajo de 1e-5, toda la suite usa 1e-5.

**Reglas que valen para todas las corridas:**

1. **Split congelado.** Se usa el `split.csv` actual para todo y no se recalcula aunque cambie la definición de ventana (A4, A5).
2. **Mismo protocolo.** Cada configuración cambia solo su campo; todo lo demás es lo de A0, incluidos los LR de A10 (ConvNeXtV2-Base).
3. **3 semillas** (42, 43, 44) en las 14 configuraciones.
4. **Test una sola vez.** Checkpoint, k de MIL y umbrales se eligen en validación; test se evalúa al final sin volver a ajustar nada.
5. **Predicciones guardadas.** Cada corrida guarda las probabilidades por ventana de validación y test; M1–M12 y el bootstrap trabajan solo sobre ellas.
6. **Trazabilidad.** `run.yaml`, `history.csv`, `metrics_test.json`; en wandb `abl-<id>-s<semilla>` (p. ej. `abl-A2-s43`).

## Matriz de entrenamiento (A0–A12)

Son 14 configuraciones × 3 semillas = 42 entrenamientos. Cada fila cambia un solo campo respecto de A0, marcado en negrita.

| ID | Pregunta | Backbone | Módulo temporal | Capas | `seq_len` / `stride` | Congelamiento | Label smoothing | Resolución |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | Referencia | ConvNeXtV2-Femto | Transformer | 1 | 16 / 16 | Parcial | 0.20 | 160 |
| A1 | ¿Hace falta modelar el tiempo? | ConvNeXtV2-Femto | **Ninguno (promedio)** | — | 16 / 16 | Parcial | 0.20 | 160 |
| A2 | ¿Transformer o recurrente? | ConvNeXtV2-Femto | **BiLSTM** | 1 | 16 / 16 | Parcial | 0.20 | 160 |
| A3 | ¿ConvNeXt o MobileViT al mismo cómputo? | **MobileViT-XS** | Transformer | 1 | 16 / 16 | Parcial | 0.20 | 160 |
| A4 | ¿Menos contexto (≈1.2 s)? | ConvNeXtV2-Femto | Transformer | 1 | **8 / 8** | Parcial | 0.20 | 160 |
| A5 | ¿Más contexto (≈5 s)? | ConvNeXtV2-Femto | Transformer | 1 | **32 / 32** | Parcial | 0.20 | 160 |
| A6 | ¿Hace falta ajustar el backbone? | ConvNeXtV2-Femto | Transformer | 1 | 16 / 16 | **Congelado** | 0.20 | 160 |
| A7 | ¿Conviene ajustarlo entero? | ConvNeXtV2-Femto | Transformer | 1 | 16 / 16 | **Descongelado** | 0.20 | 160 |
| A8 | ¿Menos suavizado? | ConvNeXtV2-Femto | Transformer | 1 | 16 / 16 | Parcial | **0.10** | 160 |
| A9 | ¿Sin suavizado? | ConvNeXtV2-Femto | Transformer | 1 | 16 / 16 | Parcial | **0.00** | 160 |
| A10 | ¿Ayuda escalar el backbone? | **ConvNeXtV2-Base** | Transformer | 1 | 16 / 16 | Parcial | 0.20 | 160 |
| A11 | ¿Más profundidad temporal? | ConvNeXtV2-Femto | Transformer | **2** | 16 / 16 | Parcial | 0.20 | 160 |
| A12 | ¿Aún más? | ConvNeXtV2-Femto | Transformer | **3** | 16 / 16 | Parcial | 0.20 | 160 |
| A13 | ¿Cuánto cuesta reducir la resolución? | ConvNeXtV2-Femto | Transformer | 1 | 16 / 16 | Parcial | 0.20 | **224** |

Comunes a todas: 5 clases, `frame_stride` 4, máx. 70 ventanas/video, 4 cabezas de atención, entropía cruzada con `WeightedRandomSampler`, y el optimizador y scheduler de A0.

### Notas por configuración

- **A1.** Promedio de las 16 características por fotograma; no hay parámetros temporales.
- **A2 (ajustes del LSTM).** BiLSTM de 1 capa con 192 unidades por dirección, así la salida tiene 384 dimensiones como el transformer. Se promedian las salidas en el tiempo, igual que el transformer, en vez de tomar el último estado. Lleva dropout 0.1 a la salida, no usa embedding posicional porque el LSTM ya es sensible al orden, y usa el grupo de LR temporal. Tiene 0.89 M parámetros frente a 1.18 M de una capa de transformer. Se elige bidireccional porque el encoder del transformer también ve toda la ventana; un LSTM unidireccional de 384 unidades igualaría parámetros, pero solo vería el pasado. El recorte de gradiente a 1.0 lo protege de gradientes explosivos con el LR mayor.
- **A3.** Backbone del modelo M-2 reportado. Usa su propio congelamiento parcial (`stem`, `stages.0`, `stages.1`). Con la misma dimensión 384, el módulo temporal y la cabeza son idénticos a A0.
- **A4 y A5.** `stride` cuenta fotogramas fuente, así que el solapamiento se mantiene en ≈73 % (lapso de 29, 61 o 125 fotogramas con paso 8, 16 o 32). Cambia el número de ventanas por video (A4 ≈ el doble, A5 ≈ la mitad) y con él los pasos por época. El tope de 70 ventanas submuestra de forma uniforme los videos largos. Ambos efectos se reportan como factores de confusión.
- **A6.** Todo el backbone congelado (`freeze_prefixes: [""]`). Hay que comprobar que el grupo de parámetros vacío del optimizador no falle.
- **A7.** Nada congelado, con el mismo LR de backbone (3e-5).
- **A10.** ConvNeXtV2-Base con el protocolo de A0: 18× parámetros y unas 20× operaciones por fotograma; el transformer pasa a 1024 dimensiones (8.4 M). Si la ganancia de AUC frente a A0 es pequeña, respalda indirectamente la compresión de la etapa 3. El profesor original no congelaba el `stem` porque el prefijo `conv_stem` no existe en ConvNeXt; A10 usa la definición de congelamiento parcial de esta suite.
- **A11 y A12.** Cada capa adicional suma 1.18 M parámetros.
- **A13.** A0 a 224×224, la resolución de preentrenamiento de ConvNeXtV2 y la del dataset; unas 2× operaciones (0.78 GMACs por fotograma). Mide cuánto AUC cuesta trabajar a 160×160: si la ganancia es pequeña, 160 queda justificado como compromiso para el borde, con un número detrás. Usa los LR y el batch de A0. Antes de lanzarla hay que confirmar que los clips exportados miden al menos 224 px de lado; si no, no tiene sentido.

## Matrices de puntuación por video (M1–M12, sin reentrenar)

Se aplican a las predicciones guardadas de las 3 semillas de A0; ninguna reentrena ni repite la inferencia. La puntuación binaria (¿es anómalo?) y la multiclase (¿qué delito?) son operaciones separadas en `lib/mil.py` y se ablacionan por separado.

| Matriz | Cambia el AUC-Macro por video | Cambia el AUC binario por video | Cambia F1 y exactitud |
| --- | --- | --- | --- |
| M1–M5: agregación de anomalía | No | Sí | Sí |
| M6–M8: umbral | No | No | Sí |
| M9–M11: agregación multiclase | Sí | No | Sí |
| M12: decisión dura actual | No | No | Sí |

En el código actual, el AUC-Macro por video ya sale del promedio de todas las ventanas (M9). Top-k y umbral solo afectan al AUC binario y a la predicción dura.

### Agregación de anomalía (M1–M5)

Puntaje de anomalía por ventana: 1 − P(Normal). Umbral 0.50.

| ID | Agregación | Fracción top-k |
| --- | --- | --- |
| M1 | Promedio de todas las ventanas | 1.00 |
| M2 | Máximo | 1 ventana |
| M3 | Promedio top-k | 0.05 |
| M4 | Promedio top-k (actual) | 0.10 |
| M5 | Promedio top-k | 0.20 |

### Umbral (M6–M8)

Promedio top-k con fracción 0.10. M7 duplica a M4 a propósito, como referencia. El umbral es un punto de operación: no cambia ningún AUC, así que se reporta con F1 y exactitud y se elige en validación.

| ID | Umbral |
| --- | --- |
| M6 | 0.40 |
| M7 | 0.50 (actual) |
| M8 | 0.60 |

### Agregación multiclase (M9–M11)

Se promedian los vectores completos de 5 clases de las ventanas elegidas. En M10 y M11, las ventanas se eligen por 1 − P(Normal).

| ID | Ventanas usadas | Agregación |
| --- | --- | --- |
| M9 | Todas (actual) | Promedio de probabilidades |
| M10 | 10 % más anómalas | Promedio de probabilidades |
| M11 | 20 % más anómalas | Promedio de probabilidades |

### Decisión dura de referencia (M12)

Es la regla actual: el video es anómalo si el promedio top-k (0.10) supera 0.50, y su clase sale de un voto por mayoría entre esas k ventanas, sin que Normal vote; los empates se resuelven por probabilidad media. Se compara su F1-Macro y exactitud por video con el argmax de M9–M11, que asigna Normal cuando el video no supera el umbral.

## Métricas y análisis estadístico

La métrica principal es el AUC-Macro por video (uno contra el resto), y cada variante se reporta como diferencia (Δ) frente a A0 con su intervalo de confianza. El test de referencia tiene solo 58 videos (678 ventanas), así que sin intervalos las diferencias pequeñas no son interpretables.

| Métrica | Nivel | Rol |
| --- | --- | --- |
| AUC-Macro | Video | Principal |
| AUC-Macro | Ventana | Secundaria, más estable por tener más muestras |
| AUC binario Normal vs. delito | Video y ventana | Secundaria; principal para M1–M5 |
| AUC por clase | Video | Control: piso de 0.65 por clase, foco en Shooting |
| F1-Macro y exactitud | Video | Secundaria; principal para M6–M8 y M12 |
| Parámetros y GFLOPs por ventana | Modelo | Todas; decisiva para A3, A10, A11 y A12 |

No se reporta latencia: la medida en la GPU de entrenamiento no representa el borde.

**Análisis:**

1. **Media ± desviación estándar** entre las 3 semillas para cada métrica.
2. **Bootstrap pareado por video:** 2 000 remuestreos de los videos de test, los mismos para A0 y la variante, sobre las probabilidades promediadas de las 3 semillas. Da el IC 95 % de Δ AUC-Macro. A nivel de ventana se remuestrean videos completos con todas sus ventanas, porque las ventanas de un mismo video no son independientes.
3. **Regla de decisión:** un componente "aporta" si Δ medio > 1 desviación estándar entre semillas **y** el IC 95 % de Δ excluye 0. Si solo se cumple una de las dos condiciones, se reporta como "tendencia" y no se afirma.
4. **AUC por clase:** solo descriptivo, porque algunas clases tienen muy pocos videos en test. Antes de empezar se publica el conteo de videos por clase en test.
5. **Validación:** se reportan también las métricas de validación, para mostrar que el ranking de variantes no depende solo del test.

Formato de la tabla del informe:

| ID | AUC-Macro video (media ± de) | Δ vs. A0 \[IC 95 %\] | AUC-Macro ventana | AUC Shooting | Parámetros (M) | Veredicto |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | — | — | — | — | 6.0 | referencia |
| A1 | — | — | — | — | — | aporta / tendencia / no aporta |

## Plan de corridas y presupuesto

Son 42 entrenamientos de la matriz más 2 cortos de piloto. Las 12 evaluaciones M1–M12 no usan GPU de forma apreciable. A10 (ConvNeXtV2-Base) dominará el tiempo total.

| Bloque | Configuraciones | Semillas | Corridas | Cómputo por fotograma vs. A0 |
| --- | --- | --- | --- | --- |
| Piloto de LR (10 épocas, solo validación) | 2 | 1 | 2 | 1× |
| A0–A9, A11, A12 | 12 | 3 | 36 | ≈1× (A3: 0.95×) |
| A13 (224×224) | 1 | 3 | 3 | ≈2× |
| A10 (ConvNeXtV2-Base) | 1 | 3 | 3 | ≈20× |
| M1–M12 (sobre A0) | 12 | — | 0 | — |
| **Total** |  |  | **44** |  |

Orden de ejecución en los dos servidores:

1. **Piloto de LR.** Fija el LR del backbone para toda la suite.
2. **A0, 3 semillas.** Mide el tiempo por corrida (t) y confirma que converge en 30 épocas o menos. Con sus predicciones se cierran M1–M12.
3. **A10 en el segundo servidor**, en cuanto el piloto termine: es la corrida más larga y no depende del resto.
4. **A1, A2, A3, A11, A12** (módulo temporal y backbone), y después **A4–A9**, en el primer servidor. A13 va en el segundo servidor, después de A10.

Ambos servidores usan el mismo entorno (versiones de PyTorch, timm y CUDA fijadas), y el `run.yaml` de cada corrida registra el servidor. Las 3 semillas de una configuración corren en el mismo servidor. Si falta tiempo, se recorta en este orden: A12, A9, A7 y A13 (12 corridas menos).

## Requisitos de implementación y preguntas abiertas

Casi todas las configuraciones son cambios de YAML con `--set` (`backbone`, `seq_len`, `stride`, `freeze_prefixes`, `label_smoothing` y `attn_layers` ya existen). Lo que falta en `master` va en una rama aparte, sin cambiar el comportamiento actual cuando el interruptor nuevo está en su valor por defecto.

| Necesidad | Dónde | Configuraciones |
| --- | --- | --- |
| `temporal_module`: `transformer` \| `none` \| `bilstm` | `lib/config.py`, `lib/model.py` | A1, A2 |
| BiLSTM: 192 unidades por dirección, promedio temporal, dropout de salida | `lib/model.py` | A2 |
| Warmup lineal de 1 época (por paso) y recorte de gradiente a 1.0 | `lib/training.py` | todas |
| Umbral absoluto en `ReduceLROnPlateau` (`threshold_mode="abs"`) | `lib/training.py` | todas |
| Acumulación de gradiente (`grad_accum`) | `lib/training.py` | las que no quepan en 16 GB |
| Guardar las probabilidades por ventana de test y val (video, fuente, clase, probabilidades) | `2_evaluate.py` | M1–M12, bootstrap |
| Agregaciones M1–M12 como funciones separadas: anomalía, multiclase y decisión dura | `lib/mil.py` | M1–M12 |
| Script de matriz de corridas y script de resumen con bootstrap pareado | nuevos, en `2-modelling/` | todas |

El congelamiento parcial de ConvNeXtV2 debe escribirse explícito como `[stem, stages.0, stages.1]`: el valor por defecto `conv_stem` no existe en ConvNeXt y por eso el `stem` del profesor nunca se congeló.

**Preguntas abiertas:**

- [ ] **Curriculum learning.** Mis notas dicen que el entrenamiento binario → multiclase está implementado, pero no está en `master`. Si existe en otra rama, se añade como A13: con curriculum vs. multiclase directo.
- [ ] **Tamaño del test.** Los 58 videos son de la corrida `wa4jsu91`. Hay que confirmar el conteo por clase del test actual, después del relleno con UCF-Crime de octubre.
- [ ] **Resultado del piloto de LR** (1e-5 o 3e-5 para el backbone).
- [ ] **Tiempo por corrida de A0 y de A10.** Decide si cabe todo antes de la entrega o si se aplican los recortes.
