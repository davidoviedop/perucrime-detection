# Etapa 5 — Ablación

Mide cuánto aporta cada pieza del modelo quitándola o cambiándola **de a una** y reentrenando
con todo lo demás igual. Reutiliza el modelo, los datos, el bucle de entrenamiento, la
puntuación MIL y las métricas de la etapa 2 (`2-modelling/lib`); aquí solo se define qué
variantes correr y cómo juntar sus resultados.

```
 config/ablation_mobilevit_xs.yaml  ──►  1_run_ablation.py  ──►  outputs/{suite}/{variante}_s{semilla}/
   (referencia + variantes)               (entrena y evalúa)            best.pt · metrics_*.json · preds_*.npz
                                                                               │
                                          2_report.py  ◄──────────────────────┤  tabla por variante (± semillas, Δ vs referencia)
                                          3_scoring_ablation.py  ◄────────────┘  re-puntúa los videos sin reentrenar
```

## Qué se ablaciona

El modelo es el de la etapa 2 con backbone **MobileViT-XS** (`2-modelling/config/teacher_mobilevit-xs.yaml`),
entrenado directamente y sin destilar: es el que permite comparar piezas a un costo razonable
(el efecto de la destilación está en la etapa 3). Cada variante cambia una sola cosa:

| Variante | Qué cambia | Pregunta que responde |
|---|---|---|
| `reference` | nada | punto de comparación |
| `temporal_mean` | sin módulo temporal: promedio de las características de los 16 fotogramas | ¿hace falta modelar el tiempo? |
| `temporal_lstm` | LSTM de 2 capas × 256 en lugar del transformer | ¿atención o recurrencia? |
| `temporal_nopos` | transformer sin embedding posicional | ¿importa el **orden** de los fotogramas? |
| `attn_layers_2` | 2 capas de atención en lugar de 1 | ¿más profundidad ayuda? |
| `unfrozen_backbone` | no se congela ninguna etapa del backbone | ¿sirve congelar las primeras etapas? |
| `scratch_backbone` | sin pesos de ImageNet (sin congelar, LR de backbone 1e-4) | ¿cuánto aporta el preentrenamiento? |
| `no_label_smoothing` | `label_smoothing: 0` | ¿ayuda el suavizado de etiquetas? |
| `uniform_sampler` | orden aleatorio uniforme en lugar del muestreo por frecuencia inversa | ¿cuánto aporta balancear las clases? |
| `no_temporal_jitter` | `temporal_jitter: 0` | ¿ayuda el desplazamiento aleatorio de la ventana? |
| `short_window` | `seq_len: 8` en lugar de 16 | ¿cuánto contexto temporal hace falta? |

Las variantes se declaran en el YAML (`model:` para la arquitectura, `set:` para cualquier campo de
`Config`, `sampler:` para el muestreador); agregar una es agregar un bloque. Al cambiar `seq_len`
cambia también el número de ventanas por video, así que `short_window` no es exactamente comparable
a nivel de ventana (a nivel de video sí).

Además, `3_scoring_ablation.py` ablaciona la **puntuación por video** sin reentrenar nada:
distintos `k` y umbrales de la puntuación MIL, voto por mayoría de todas las ventanas, probabilidad
media, max-pooling y, con `--shuffle-frames`, el mismo modelo con el orden de los fotogramas
barajado.

## Estructura

```
5-ablation/
├── 1_run_ablation.py       Paso 1  entrena y evalúa cada (variante, semilla) → outputs/{suite}/
├── 2_report.py             Paso 2  tabla de resultados → results/ablation_{suite}.csv y .md
├── 3_scoring_ablation.py   Paso 3  ablación de la puntuación por video (sin entrenar)
├── config/                 ablation_mobilevit_xs.yaml: la suite (configuración base + variantes)
├── results/                tablas generadas (CSV + Markdown)
├── requirements.txt
└── ablation/               lógica propia de la etapa (no se llama `lib` para no chocar con la etapa 2)
    ├── variants.py         módulos temporales alternativos (mean, lstm, transformer sin posición)
    ├── suite.py            carga la suite, entrena y evalúa cada corrida, guarda métricas y predicciones
    ├── mil_variants.py     formas alternativas de puntuar un video
    └── report.py           agrega las corridas: media ± desviación y Δ contra la referencia
```

## Uso

```bash
python 1_run_ablation.py --list                          # muestra las corridas y qué cambia cada una; no corre nada
python 1_run_ablation.py --only reference temporal_mean  # solo algunas variantes (la referencia siempre se incluye)
python 1_run_ablation.py --seeds 42 43 44                # varias semillas por variante
python 1_run_ablation.py --set epochs=10                 # recorta el entrenamiento (prueba rápida)
python 1_run_ablation.py                                 # toda la suite
python 2_report.py
python 3_scoring_ablation.py --run-dir outputs/mobilevit_xs/reference_s42 [--shuffle-frames]
```

* Todos los comandos se ejecutan desde esta carpeta (`5-ablation/`). Los datos salen de `data_root`
  de la configuración base (`--data-root` o `--set data_root=...` lo cambian).
* **Se puede interrumpir y repetir el comando:** una corrida con `metrics_test.json` se da por
  terminada y se omite; una con `last.pt` se reanuda; una con `best.pt` pero sin métricas solo se
  evalúa. `--retrain` fuerza repetirlas. Si una variante falla, las demás siguen y al final se lista
  cuáles fallaron.
* Cada corrida guarda, además de lo de la etapa 2, `metrics_val.json`, `metrics_test.json` (mismas
  métricas que `2_evaluate.py`), `preds_{val,test}.npz` (probabilidades por ventana) y `variant.json`
  (qué se cambió, parámetros, época del mejor modelo).
* El mejor modelo de cada corrida se elige, como en la etapa 2, por el AUC de **validación**; la
  tabla reporta **test**.

## Cómo leer los resultados (importante)

* **El test es pequeño:** 58 videos, con 4 de Robbery y 4 de Shooting. Un solo video cambia el
  recall de esas clases en 25 puntos. Diferencias de uno o dos puntos de AUC o F1 entre variantes
  están dentro del ruido; para concluir algo hay que repetir con varias semillas (`--seeds`) y mirar
  la desviación que reporta `2_report.py`.
* Con una sola semilla (el valor por defecto de la suite) los Δ contra la referencia son
  **indicativos, no concluyentes**.
* **Una ablación quita una pieza; no prueba que las demás sean óptimas.** Los LR y demás
  hiperparámetros son los de la referencia y no se reajustan por variante. En `scratch_backbone` se
  sube el LR del backbone y se descongela porque entrenar desde cero con el LR de ajuste fino no
  tiene sentido, pero aun así no es un entrenamiento afinado.
* El modelo ablacionado es MobileViT-XS entrenado directamente; el modelo final del trabajo es el
  estudiante destilado de la etapa 3, de modo que las conclusiones valen para la arquitectura y el
  entrenamiento, no necesariamente para la destilación.

## Qué se cambió en otras etapas

Solo `2-modelling/lib/pipeline.py`: `train_run` recibe dos ganchos opcionales,
`model_factory` (otro constructor de modelo) y `loaders_hook` (modificar los loaders antes de
entrenar). Ambos valen `None` por defecto, así que las etapas 2 y 3 se comportan igual que antes.

## Cómo se comprobó

* **Prueba de extremo a extremo** con los datos reales y una configuración mínima (1 época, 3 ventanas
  por video, sin pesos preentrenados): `reference`, `temporal_lstm` y `uniform_sampler` entrenan, se
  evalúan en val y test y dejan todos sus archivos; `2_report.py` arma la tabla; `3_scoring_ablation.py`
  re-puntúa la corrida (también con `--shuffle-frames`); al repetir el comando las corridas terminadas
  se omiten. Las cifras de esa prueba no significan nada (1 época desde cero); solo valida el flujo.
* **Misma evaluación que la etapa 2:** las 54 métricas de test que calcula `ablation/suite.py` para la
  corrida de referencia coinciden exactamente con las de `2-modelling/2_evaluate.py` sobre la misma corrida.
* **No se ha corrido la suite completa**: con 40 épocas por variante y 11 variantes (más las semillas
  adicionales) es un entrenamiento largo; lánzalo con `--list` primero para ver qué se va a correr.
