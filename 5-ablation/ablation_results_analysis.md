# Ablación de modelado — resultados y análisis

Nov 24, 2026 · suite `modelling` (A0–A13, 3 semillas) y matrices de puntuación por video (M1–M12)

Este documento reúne **todos los resultados** de la ablación definida en [`ablation_definition.md`](ablation_definition.md), su análisis y los pasos siguientes. Las tablas salen de `results/ablation_modelling.md` y `results/scoring_M1-M12.md` (generadas por `2_report.py` y `3_scoring_ablation.py`); los detalles por corrida están en `reports/` y `outputs/modelling/`.

## 1. Resumen ejecutivo

* **Se completaron las 42 corridas** (14 configuraciones × 3 semillas) sin fallas, en 31.7 h en un solo servidor (RTX 4070 Ti SUPER). Todas pararon por parada temprana; ninguna llegó a las 30 épocas.
* **Solo una configuración cumple la regla de decisión de «aporta»:** **A13**, entrenar y evaluar a **224×224** en lugar de 160×160 (Δ AUC-Macro por video **+0.066** [+0.013, +0.124]), con 2× el cómputo. Pero la validación **no lo confirma** (0.833 vs. 0.839), así que el hallazgo queda como probable, no probado.
* **Tendencias** (una sola de las dos condiciones): **A10** escalar a ConvNeXtV2-Base ↑ (+0.048, con 16× los parámetros y 20× el cómputo); **A6** congelar todo el backbone ↓ (−0.042); **A9** quitar el label smoothing ↓ (−0.025); **A5** ventanas de ≈5 s ↓ (−0.026, sobre los 45 videos que puede puntuar).
* **Todo lo demás no se distingue del ruido:** quitar el módulo temporal (A1), cambiarlo por un BiLSTM (A2), MobileViT en lugar de ConvNeXt (A3), ventana corta (A4), descongelar todo (A7), menos label smoothing (A8) y más capas temporales (A11, A12). **El módulo temporal no demuestra aportar** (A1 ≥ A0) **y el transformer no demuestra superar a un LSTM** (A2 ≈ A0).
* **La puntuación por video (M1–M12) no es un componente sensible:** top-k, máximo o promedio dan el mismo AUC binario, y los umbrales 0.4–0.6 dan resultados idénticos.
* **La evidencia es de baja resolución:** los IC 95 % miden en promedio ±0.053 de AUC; las clasificaciones de validación y test casi no coinciden (Spearman 0.21); el AUC de entrenamiento llega a 1.0 mientras el de validación se queda en ≈0.79 y el mejor modelo sale de la época 1–3 en 29 de 42 corridas. Efectos menores de ≈0.05 no se pueden detectar con este test (58 videos).
* **El piloto de LR del diseño no se corrió:** todas las corridas usaron `lr_backbone = 3e-5` sin la comprobación contra 1e-5 (sección 6).

## 2. Qué se ejecutó

| Aspecto | Valor |
|---|---|
| Suite | `config/ablation_modelling.yaml`; base A0 en `config/a0_convnextv2_femto.yaml` |
| Modelo base A0 | ConvNeXtV2-Femto (ImageNet, congelamiento parcial `stem`, `stages.0`, `stages.1`) + transformer temporal de 1 capa y 4 cabezas + cabeza lineal; 6.04 M parámetros, 12.7 GFLOPs por ventana |
| Datos | `data/peru-crime5`: 5 clases (Assault, Normal, Road_Accident, Robbery, Shooting); test de 58 videos / 678 ventanas (Assault 14, Normal 29, Road_Accident 7, Robbery 4, Shooting 4). Ventanas de 16 fotogramas (uno de cada cuatro), 160×160 |
| Entrenamiento | AdamW; LR 3e-5 (backbone) / 3e-4 (temporal y cabeza); warmup de 1 época; `ReduceLROnPlateau` (patience 1, umbral absoluto 1e-3); máx. 30 épocas, parada temprana con patience 5 sobre el AUC de validación; label smoothing 0.2 |
| Semillas | 42, 43, 44 en todas las configuraciones |
| Entorno | un servidor (`COSDEVAI`), torch 2.14.1, timm 1.0.30, GPU de 16 GB; wandb activado |
| Tiempo | 31.7 h en total; ≈25 min por corrida salvo A4 (52), A6 (61), A5 (16) y **A10 (215 min)** |
| Análisis | media ± desviación entre semillas; Δ frente a A0 con IC 95 % por bootstrap pareado por video (2 000 remuestreos estratificados por clase sobre las probabilidades promediadas entre semillas); regla: «aporta» si |Δ| supera la mayor desviación entre semillas **y** el IC excluye 0, «tendencia» si se cumple solo una |

Las 14 configuraciones, cada una cambia un solo campo de A0: A1 sin módulo temporal; A2 BiLSTM; A3 MobileViT-XS; A4 `seq_len` 8; A5 `seq_len` 32; A6 backbone congelado; A7 sin congelar; A8 label smoothing 0.10; A9 sin label smoothing; A10 ConvNeXtV2-Base; A11 y A12 2 y 3 capas de atención; A13 resolución 224.

## 3. Resultados de entrenamiento (test)

### 3.1 Tabla principal

AUC-Macro uno-contra-resto por video (la métrica principal). Δ frente a A0 con IC 95 %.

| ID | Pregunta | AUC-Macro video | Δ vs A0 [IC 95 %] | AUC-Macro ventana | AUC Shooting | Parám. (M) | GFLOPs/ventana | Veredicto |
|---|---|---|---|---|---|---|---|---|
| A0 | Referencia | 0.766 ± 0.012 | — | 0.710 ± 0.042 | 0.583 ± 0.089 | 6.04 | 12.7 | referencia |
| A1 | ¿Hace falta modelar el tiempo? | 0.774 ± 0.037 | +0.022 [−0.034, +0.075] | 0.739 ± 0.037 | 0.606 ± 0.157 | 4.85 | 12.7 | no aporta |
| A2 | ¿Transformer o recurrente? | 0.768 ± 0.031 | +0.010 [−0.024, +0.045] | 0.756 ± 0.009 | 0.657 ± 0.082 | 5.74 | 12.7 | no aporta |
| A3 | ¿ConvNeXt o MobileViT al mismo cómputo? | 0.758 ± 0.013 | +0.003 [−0.059, +0.064] | 0.748 ± 0.033 | 0.497 ± 0.137 | 3.13 | 12.1 | no aporta |
| A4 | ¿Menos contexto (≈1.2 s)? | 0.743 ± 0.041 | −0.014 [−0.067, +0.040] | 0.722 ± 0.023 | 0.489 ± 0.138 | 6.04 | 6.4 | no aporta |
| A5 | ¿Más contexto (≈5 s)? | 0.793 ± 0.032 * | −0.026 [−0.091, +0.040] * | 0.711 ± 0.051 | 0.707 ± 0.058 | 6.05 | 25.4 | tendencia ↓ |
| A6 | ¿Hace falta ajustar el backbone? | 0.729 ± 0.020 | −0.042 [−0.106, +0.022] | 0.670 ± 0.006 | 0.685 ± 0.024 | 6.04 | 12.7 | tendencia ↓ |
| A7 | ¿Conviene ajustarlo entero? | 0.762 ± 0.056 | −0.006 [−0.032, +0.024] | 0.697 ± 0.027 | 0.622 ± 0.234 | 6.04 | 12.7 | no aporta |
| A8 | ¿Menos suavizado? | 0.770 ± 0.032 | +0.005 [−0.015, +0.027] | 0.729 ± 0.035 | 0.614 ± 0.131 | 6.04 | 12.7 | no aporta |
| A9 | ¿Sin suavizado? | 0.739 ± 0.005 | −0.025 [−0.075, +0.025] | 0.708 ± 0.030 | 0.506 ± 0.007 | 6.04 | 12.7 | tendencia ↓ |
| A10 | ¿Ayuda escalar el backbone? | 0.804 ± 0.034 | +0.048 [−0.008, +0.109] | 0.780 ± 0.031 | 0.702 ± 0.134 | 96.12 | 250.7 | tendencia ↑ |
| A11 | ¿Más profundidad temporal? | 0.760 ± 0.022 | −0.021 [−0.070, +0.029] | 0.727 ± 0.022 | 0.583 ± 0.135 | 7.22 | 12.7 | no aporta |
| A12 | ¿Aún más? | 0.727 ± 0.053 | −0.045 [−0.100, +0.004] | 0.670 ± 0.082 | 0.576 ± 0.090 | 8.41 | 12.7 | no aporta |
| A13 | ¿Cuánto cuesta reducir la resolución? | 0.816 ± 0.009 | **+0.066 [+0.013, +0.124]** | 0.784 ± 0.005 | 0.730 ± 0.110 | 6.04 | 24.9 | **aporta ↑** |

\* **A5 solo puntúa 45 de los 58 videos de test** (los clips más cortos que su ventana de ≈5 s no producen ninguna ventana: 9 Normal, 1 Road_Accident, 1 Robbery y 2 de VioPeru). Su AUC de 0.793 es sobre esos 45 videos; su Δ, IC y veredicto se calculan sobre los 45 videos comunes, restringiendo también a A0 (en ese subconjunto A0 vale 0.825 frente a 0.793 de A5). La columna de AUC de A5 no es comparable con las demás.

**AUC por semilla** (test, video), para ver el solapamiento entre configuraciones:

| ID | s42 | s43 | s44 | ID | s42 | s43 | s44 |
|---|---|---|---|---|---|---|---|
| A0 | 0.781 | 0.758 | 0.761 | A7 | 0.800 | 0.788 | 0.698 |
| A1 | 0.795 | 0.732 | 0.797 | A8 | 0.803 | 0.766 | 0.741 |
| A2 | 0.767 | 0.799 | 0.737 | A9 | 0.745 | 0.736 | 0.735 |
| A3 | 0.760 | 0.744 | 0.770 | A10 | 0.833 | 0.766 | 0.813 |
| A4 | 0.784 | 0.701 | 0.744 | A11 | 0.768 | 0.776 | 0.735 |
| A5 | 0.758 | 0.820 | 0.801 | A12 | 0.772 | 0.668 | 0.741 |
| A6 | 0.707 | 0.745 | 0.733 | A13 | 0.826 | 0.815 | 0.807 |

(A5 sobre sus propios videos.) Las tres semillas de A13 quedan por encima de las tres de A0 (0.807–0.826 frente a 0.758–0.781); en A10, dos de tres.

### 3.2 Métricas secundarias (test, video)

| ID | AUC binario | F1-Macro | Exactitud | Δ AUC-Macro ventana [IC 95 %] |
|---|---|---|---|---|
| A0 | 0.788 ± 0.033 | 0.364 ± 0.070 | 0.523 ± 0.020 | — |
| A1 | 0.803 ± 0.021 | 0.404 ± 0.036 | 0.575 ± 0.036 | +0.023 [−0.027, +0.085] |
| A2 | 0.803 ± 0.015 | 0.409 ± 0.063 | 0.546 ± 0.070 | +0.027 [−0.009, +0.052] |
| A3 | 0.784 ± 0.058 | 0.339 ± 0.102 | 0.477 ± 0.105 | +0.031 [−0.035, +0.100] |
| A4 | 0.825 ± 0.057 | 0.323 ± 0.015 | 0.506 ± 0.036 | −0.008 [−0.057, +0.044] |
| A5 | 0.787 ± 0.027 | 0.421 ± 0.112 | 0.578 ± 0.059 | −0.027 [−0.096, +0.067] |
| A6 | 0.714 ± 0.024 | 0.408 ± 0.028 | 0.523 ± 0.010 | −0.058 [−0.106, +0.031] |
| A7 | 0.801 ± 0.021 | 0.341 ± 0.076 | 0.506 ± 0.055 | −0.036 [−0.058, +0.006] |
| A8 | 0.774 ± 0.045 | 0.382 ± 0.097 | 0.540 ± 0.026 | +0.007 [−0.009, +0.023] |
| A9 | 0.753 ± 0.047 | 0.333 ± 0.071 | 0.477 ± 0.061 | −0.014 [−0.057, +0.026] |
| A10 | 0.755 ± 0.039 | 0.413 ± 0.023 | 0.517 ± 0.017 | +0.057 [+0.003, +0.113] |
| A11 | 0.819 ± 0.019 | 0.406 ± 0.039 | 0.552 ± 0.062 | −0.012 [−0.072, +0.043] |
| A12 | 0.703 ± 0.094 | 0.376 ± 0.051 | 0.500 ± 0.062 | −0.089 [−0.136, +0.001] |
| A13 | 0.788 ± 0.015 | 0.374 ± 0.053 | 0.557 ± 0.043 | +0.077 [+0.030, +0.126] |

### 3.3 AUC por clase (test, video; solo descriptivo)

Con 4 videos de test en Robbery y Shooting (7 en Road_Accident) estas cifras son muy ruidosas.

| ID | Assault | Normal | Road_Accident | Robbery | Shooting |
|---|---|---|---|---|---|
| A0 | 0.96 ± 0.01 | 0.79 ± 0.03 | 0.70 ± 0.01 | 0.80 ± 0.06 | 0.58 ± 0.09 |
| A1 | 0.98 ± 0.01 | 0.80 ± 0.02 | 0.68 ± 0.02 | 0.81 ± 0.03 | 0.61 ± 0.16 |
| A2 | 0.98 ± 0.00 | 0.79 ± 0.02 | 0.66 ± 0.05 | 0.75 ± 0.05 | 0.66 ± 0.08 |
| A3 | 0.94 ± 0.02 | 0.77 ± 0.05 | 0.72 ± 0.03 | 0.85 ± 0.03 | 0.50 ± 0.14 |
| A4 | 0.98 ± 0.00 | 0.82 ± 0.07 | 0.66 ± 0.06 | 0.77 ± 0.10 | 0.49 ± 0.14 |
| A5 | 0.95 ± 0.02 | 0.80 ± 0.02 | 0.68 ± 0.07 | 0.82 ± 0.08 | 0.71 ± 0.06 |
| A6 | 0.95 ± 0.01 | 0.71 ± 0.03 | 0.63 ± 0.04 | 0.67 ± 0.09 | 0.69 ± 0.02 |
| A7 | 0.96 ± 0.02 | 0.80 ± 0.02 | 0.69 ± 0.01 | 0.74 ± 0.08 | 0.62 ± 0.23 |
| A8 | 0.95 ± 0.00 | 0.78 ± 0.03 | 0.71 ± 0.02 | 0.79 ± 0.09 | 0.61 ± 0.13 |
| A9 | 0.95 ± 0.01 | 0.75 ± 0.03 | 0.70 ± 0.04 | 0.80 ± 0.04 | 0.51 ± 0.01 |
| A10 | 0.96 ± 0.01 | 0.74 ± 0.05 | 0.73 ± 0.04 | 0.89 ± 0.02 | 0.70 ± 0.13 |
| A11 | 0.97 ± 0.01 | 0.81 ± 0.02 | 0.69 ± 0.02 | 0.75 ± 0.02 | 0.58 ± 0.14 |
| A12 | 0.95 ± 0.03 | 0.70 ± 0.10 | 0.69 ± 0.02 | 0.72 ± 0.09 | 0.58 ± 0.09 |
| A13 | 0.97 ± 0.00 | 0.80 ± 0.02 | 0.72 ± 0.03 | 0.85 ± 0.07 | 0.73 ± 0.11 |

### 3.4 Validación y comportamiento del entrenamiento

| ID | AUC-Macro video (val) | AUC-Macro ventana (val) | Mejor época (media) | Épocas (media) | Tiempo por corrida (min) |
|---|---|---|---|---|---|
| A0 | 0.839 ± 0.019 | 0.823 ± 0.031 | 3.3 | 8.3 | 27 |
| A1 | 0.847 ± 0.030 | 0.846 ± 0.064 | 2.7 | 7.7 | 25 |
| A2 | 0.874 ± 0.010 | 0.866 ± 0.004 | 3.0 | 8.0 | 26 |
| A3 | 0.813 ± 0.018 | 0.794 ± 0.026 | 1.7 | 6.7 | 22 |
| A4 | 0.868 ± 0.005 | 0.849 ± 0.013 | 6.0 | 11.0 | 52 |
| A5 | 0.840 ± 0.021 | 0.840 ± 0.028 | 2.3 | 7.3 | 16 |
| A6 | 0.815 ± 0.009 | 0.768 ± 0.008 | 14.0 | 19.0 | 61 |
| A7 | 0.850 ± 0.014 | 0.846 ± 0.014 | 3.7 | 8.7 | 29 |
| A8 | 0.840 ± 0.021 | 0.824 ± 0.043 | 3.0 | 8.0 | 26 |
| A9 | 0.841 ± 0.046 | 0.838 ± 0.028 | 3.3 | 8.3 | 27 |
| A10 | 0.856 ± 0.024 | 0.838 ± 0.033 | 1.7 | 6.7 | 215 |
| A11 | 0.865 ± 0.026 | 0.831 ± 0.046 | 3.7 | 8.7 | 29 |
| A12 | 0.828 ± 0.027 | 0.800 ± 0.030 | 2.7 | 7.7 | 25 |
| A13 | 0.833 ± 0.010 | 0.801 ± 0.021 | 2.3 | 7.3 | 28 |

Correlación de Spearman entre el AUC-Macro por video de validación y de test, sobre las 14 configuraciones: **0.21**.

Sobre las 42 corridas: todas pararon por parada temprana (mediana 8 épocas); **la mejor época fue ≤ 3 en 29 corridas y ≤ 5 en 34** (máximo 21, en A6); la mediana del AUC de entrenamiento en la última época fue **1.00** frente a **0.79** en validación.

## 4. Puntuación por video (M1–M12, sobre las 3 semillas de A0)

La puntuación binaria (¿es anómalo?) y la multiclase (¿qué delito?) se ablacionan por separado (`lib/mil.py`).

**Agregación de anomalía (M1–M5):** cambia el AUC binario por video.

| ID | Agregación | AUC binario test | Δ vs M4 [IC 95 %] | F1-Macro test | Exactitud test | AUC binario val |
|---|---|---|---|---|---|---|
| M1 | Promedio de todas las ventanas | 0.789 ± 0.029 | +0.014 [−0.029, +0.063] | 0.347 ± 0.027 | 0.534 ± 0.017 | 0.783 ± 0.018 |
| M2 | Máximo (1 ventana) | 0.786 ± 0.033 | +0.001 [−0.000, +0.007] | 0.364 ± 0.070 | 0.523 ± 0.020 | 0.815 ± 0.030 |
| M3 | Promedio top-k 5 % | 0.787 ± 0.033 | +0.001 [−0.000, +0.007] | 0.364 ± 0.070 | 0.523 ± 0.020 | 0.814 ± 0.031 |
| M4 | Promedio top-k 10 % (actual) | 0.788 ± 0.033 | — | 0.364 ± 0.070 | 0.523 ± 0.020 | 0.813 ± 0.029 |
| M5 | Promedio top-k 20 % | 0.789 ± 0.036 | +0.007 [+0.000, +0.020] | 0.364 ± 0.070 | 0.523 ± 0.020 | 0.808 ± 0.027 |

**Umbral (M6–M8):** no cambia ningún AUC; punto de operación. Elegido en validación (mayor F1-Macro): M7 (0.50).

| ID | Umbral | F1-Macro val | Exactitud val | F1-Macro test | Exactitud test |
|---|---|---|---|---|---|
| M6 | 0.40 | 0.461 ± 0.079 | 0.615 ± 0.065 | 0.364 ± 0.070 | 0.523 ± 0.020 |
| M7 ✓ | 0.50 (actual) | 0.462 ± 0.079 | 0.615 ± 0.065 | 0.364 ± 0.070 | 0.523 ± 0.020 |
| M8 | 0.60 | 0.462 ± 0.079 | 0.615 ± 0.065 | 0.364 ± 0.070 | 0.523 ± 0.020 |

**Agregación multiclase (M9–M11):** cambia el AUC-Macro por video.

| ID | Ventanas usadas | AUC-Macro test | Δ vs M9 [IC 95 %] | F1-Macro test | Exactitud test | AUC-Macro val |
|---|---|---|---|---|---|---|
| M9 | Promedio de todas las ventanas (actual) | 0.766 ± 0.012 | — | 0.359 ± 0.065 | 0.506 ± 0.020 | 0.839 ± 0.019 |
| M10 | Promedio del 10 % más anómalo | 0.763 ± 0.006 | −0.006 [−0.026, +0.014] | 0.355 ± 0.071 | 0.500 ± 0.030 | 0.853 ± 0.015 |
| M11 | Promedio del 20 % más anómalo | 0.760 ± 0.007 | −0.004 [−0.023, +0.015] | 0.355 ± 0.071 | 0.500 ± 0.030 | 0.849 ± 0.018 |

**Decisión dura (M12)** frente al argmax de M9–M11 (Normal si el video no supera el umbral; si lo supera, la clase no Normal más probable):

| ID | Decisión | F1-Macro test | Exactitud test | F1-Macro val | Exactitud val |
|---|---|---|---|---|---|
| M12 | Umbral + voto entre las k ventanas, sin que Normal vote (actual) | 0.364 ± 0.070 | 0.523 ± 0.020 | 0.462 ± 0.079 | 0.615 ± 0.065 |
| M9 | argmax, promedio de todas las ventanas | 0.359 ± 0.065 | 0.506 ± 0.020 | 0.456 ± 0.046 | 0.580 ± 0.020 |
| M10 | argmax, promedio del 10 % más anómalo | 0.355 ± 0.071 | 0.500 ± 0.030 | 0.431 ± 0.058 | 0.563 ± 0.020 |
| M11 | argmax, promedio del 20 % más anómalo | 0.355 ± 0.071 | 0.500 ± 0.030 | 0.426 ± 0.050 | 0.563 ± 0.020 |

## 5. Análisis

### 5.1 Conclusión por prueba

* **A13 — resolución 224 (aporta ↑, con reservas).** Es el mayor efecto y el único que pasa la regla: +0.066 en AUC-Macro por video, +0.077 por ventana, y mejora Shooting (0.73) y Robbery (0.85). Es muy estable (desviación 0.009) y sus tres semillas superan a las de A0. **Reservas:** (i) la validación no lo ve (0.833 vs. 0.839 por video; 0.801 vs. 0.823 por ventana); (ii) 6 de los 58 videos de test (todos Normal) miden menos de 224 px de lado menor y se reescalan hacia arriba; (iii) cuesta 2× el cómputo. Es una hipótesis fuerte, no una conclusión cerrada.
* **A10 — backbone Base (tendencia ↑).** +0.048 con IC que incluye 0, por 16× los parámetros y 20× el cómputo; 215 min por corrida frente a ≈25. La validación apenas cambia (+0.017) y el AUC binario baja (0.755 vs. 0.788). Mejora Robbery (0.89) y Shooting (0.70). La capacidad ayuda algo pero no es el cuello de botella: **un aumento de 2× en cómputo por resolución rinde más que uno de 20× por el backbone.**
* **A6 — backbone congelado (tendencia ↓).** La señal más consistente entre métricas: peor en test (−0.042), en validación (0.815), por ventana (0.670) y en AUC binario (0.714); además tarda 14 épocas en alcanzar su mejor punto. Ajustar el backbone ayuda.
* **A9 — sin label smoothing (tendencia ↓).** −0.025 en test, pero la validación es igual (0.841 vs. 0.839). Evidencia débil a favor del suavizado; A8 (0.10) equivale a A0 (0.20).
* **A5 — ventana de ≈5 s (tendencia ↓).** Sobre los 45 videos comunes no mejora (−0.026) y cuesta 2× el cómputo; además pierde 13 videos de test por no poder armar ventanas. No compensa.
* **A4 — ventana de ≈1.2 s (no aporta).** Sin pérdida detectable (−0.014); en validación es incluso mejor (0.868) y tiene el mayor AUC binario (0.825). Reduce a la mitad los GFLOPs por ventana (6.4) aunque produce el doble de ventanas.
* **A1 — sin módulo temporal (no aporta).** Promediar las características de los fotogramas iguala al transformer (+0.022) y es algo mejor en AUC binario, F1 y exactitud, con 1.2 M parámetros menos. **El modelado temporal no demuestra aportar en este dataset y con esta ventana.**
* **A2 — BiLSTM (no aporta).** Equivale al transformer (+0.010), es el mejor en validación (0.874) y tiene menos parámetros. **No hay evidencia de que la atención supere a la recurrencia.**
* **A3 — MobileViT-XS (no aporta).** Mismo AUC en test (+0.003) con la mitad de los parámetros (3.1 M); algo peor en validación (0.813) y en Shooting (0.50).
* **A7 — descongelar todo (no aporta).** Igual que el congelamiento parcial (−0.006).
* **A11 y A12 — más capas temporales (no aportan).** A11 −0.021; A12 −0.045 con alta variabilidad (0.727 ± 0.053). Más profundidad temporal no ayuda.
* **M1–M12 — puntuación por video.** El AUC binario es 0.786–0.789 para promedio, máximo y top-k, con Δ ≤ 0.014 y todos los IC incluyendo 0; los umbrales 0.4, 0.5 y 0.6 dan **resultados idénticos**, lo que indica que los puntajes por video son casi bimodales; promediar solo las ventanas más anómalas no mejora al promedio de todas (−0.006); la decisión dura actual (M12) es igual o algo mejor que las alternativas con argmax. El diseño de la puntuación MIL es válido pero no es una palanca.

### 5.2 Lectura de conjunto

1. **Lo que parece importar** es lo que cambia la información que llega al clasificador o cuánto se adapta el backbone: la **resolución de entrada** (A13), la **adaptación del backbone** (A6) y, en menor medida, la **capacidad** (A10). Lo que se hace *después* de extraer las características de cada fotograma (módulo temporal, profundidad, tipo de módulo, longitud de ventana, puntuación por video) no se distingue del ruido.
2. **Implicaciones sobre el modelado temporal.** El hecho de que A1 (promedio) ≥ A0, A2 ≈ A0, y que ventanas de 1.2 s y 2.4 s rindan igual sugiere que el clasificador se apoya sobre todo en la apariencia por fotograma. Es consistente con clases que se reconocen por el contenido de la escena (un arma, vehículos, un entorno) más que por la dinámica, aunque no se ha probado (ver 7.1).
3. **Por qué las diferencias de arquitectura son pequeñas** (hipótesis, por orden de apoyo en los datos): (a) el test no tiene resolución para efectos menores de ≈0.05; (b) las características preentrenadas por fotograma concentran la señal; (c) el entrenamiento memoriza 268 videos en pocas épocas (AUC de entrenamiento 1.0, mejor época 1–3), lo que deja poco margen para que el módulo temporal aprenda algo distinto; (d) las etiquetas por ventana heredan la del video aunque el incidente ocupe parte del clip, lo que impone un techo de ruido (≈0.77–0.82 de AUC) que ninguna arquitectura supera.
4. **La confiabilidad es baja en general:** IC 95 % de ±0.053 de media (entre ±0.021 y ±0.065), Spearman val/test de 0.21 y desviaciones entre semillas comparables con los Δ. Un veredicto «no aporta» significa «no detectable aquí», no «no tiene efecto».

## 6. Límites y amenazas a la validez

* **Tamaño del test:** 58 videos; 4 de Robbery y de Shooting; un solo video cambia el recall de esas clases en 25 puntos. El AUC por clase es solo descriptivo.
* **Piloto de LR no ejecutado:** el diseño exigía comparar 1e-5 y 3e-5 en A0 antes de la suite y usar 1e-5 si 3e-5 quedaba por debajo. No se corrió (`outputs/` solo tiene `modelling`), así que todo usa 3e-5 sin comprobar. Dado que A6 sugiere que la adaptación del backbone importa, el LR del backbone es una variable que puede cambiar conclusiones.
* **Un solo LR y protocolo para todas las variantes** (sin reajuste por variante) y diseño de un factor a la vez: no se evalúan **interacciones** (por ejemplo, si el módulo temporal importa a 224 px o con un backbone más fuerte).
* **A5:** pierde 13 de 58 videos de test y 12 de 58 de validación (los clips cortos no producen ventanas); es probable que también pierda videos de entrenamiento, no verificado. Su comparación solo es válida sobre los 45 videos comunes.
* **A13:** validación en desacuerdo con test; 6 clips de test menores de 224 px reescalados hacia arriba.
* **Un solo servidor:** cumple la regla de las tres semillas en el mismo servidor, pero no hay réplica en otra máquina (las semillas de CUDA no garantizan igualdad bit a bit entre GPUs).
* **Decisiones de interpretación** (documentadas en el README de la etapa): la regla de decisión usa la mayor de las dos desviaciones entre semillas; el bootstrap es estratificado por clase; la decisión de M9–M11 asigna Normal si el video no supera el umbral y, si lo supera, la clase no Normal más probable; los GFLOPs de A2 no incluyen la celda LSTM (despreciable frente al backbone).
* **Posible atajo por fuente:** la clase Assault proviene solo de VioPeru y su AUC es 0.94–0.98 en todas las variantes. No se ha comprobado cuánto depende de rasgos de esas cámaras.
* **Las conclusiones valen para el detector entrenado directamente**, no para el estudiante destilado (etapas 3 y 4), que se ablacionará aparte.

## 7. Pasos siguientes

### 7.1 Prioridad alta (baratos y aclaran lo más incierto)

1. **Correr el piloto de LR** (≈2 corridas de 10 épocas, solo validación):
   `python 1_run_ablation.py --suite config/pilot_lr.yaml` y `python 2_report.py --suite config/pilot_lr.yaml`. Si 1e-5 supera a 3e-5, repetir A0 y las configuraciones clave con ese LR.
2. **Prueba de barajado de fotogramas** sobre los checkpoints de A0 y A1: evaluar con el orden de los fotogramas de cada ventana barajado. Si las métricas no cambian, el modelo ignora el orden temporal y se confirma la hipótesis de apariencia por fotograma. (Requiere agregar esa opción a `3_scoring_ablation.py`; antes existía en una versión previa y se retiró.)
3. **Verificar A13 por separado:** (a) repetir A13 con 192 px y 256 px (2 semillas adicionales cada una) para ver si el efecto escala con la resolución; (b) recalcular A0 y A13 en test **excluyendo los 6 videos menores de 224 px**; (c) mirar por qué la validación no sigue a test. `0_check_data.py` lista los videos pequeños.
4. **Poder estadístico:** calcular el efecto mínimo detectable con los IC actuales (≈0.05) y decidir cuántas semillas o videos hacen falta para detectar un Δ de 0.02–0.03.

### 7.2 Prioridad media (mejoran la solidez de las conclusiones)

5. **Más resolución de evaluación:** evaluar con validación cruzada por video, o juntar val + test (116 videos) para las comparaciones clave, y subir a 5–10 semillas en A0, A1, A2 y A13. Un `--seeds 45 46 47 48 49` sobre esas configuraciones es suficiente con el runner actual.
6. **Protocolo v2 contra el sobreajuste** (el AUC de entrenamiento llega a 1.0 en pocas épocas): menor LR de backbone, más regularización o aumento de datos, parada por pérdida de validación, o entrenar solo la cabeza unas épocas antes de ajustar. Repetir entonces A0, A1, A2 y A13: si el sobreajuste es lo que oculta las diferencias, aparecerán.
7. **Combinaciones:** A13 + A1 y A13 + A2 para ver si el módulo temporal importa a mayor resolución (interacciones que un diseño de un factor a la vez no ve).
8. **Análisis por fuente:** AUC separado para videos locales, UCF-Crime y VioPeru, para medir el posible atajo de fuente en Assault.
9. **A5 sin pérdida de videos:** rellenar los clips cortos (repetir fotogramas) o evaluar a nivel de video con una ventana más corta en los clips que no alcanzan, para comparar sobre los 58 videos.

### 7.3 Prioridad baja o de más largo plazo

10. **Datos y etiquetas:** más videos de Robbery, Shooting y Assault local (hoy 4 videos de test por clase) y etiquetado temporal del incidente para entrenar con ventanas realmente positivas; probablemente la mejora de mayor impacto.
11. **Ablación de optimización** (etapas 3 y 4) sobre el modelo que salga de estas pruebas; en particular, decidir la resolución de trabajo según A13 antes de destilar.
12. **Repetir A1 y A2 con ConvNeXtV2-Base** (A10 mostró una ganancia posible por capacidad) solo si el paso 6 sugiere que el módulo temporal puede importar; cada corrida de Base cuesta ≈3.5 h.

### 7.4 Qué decidir ahora

* **Resolución de trabajo (160 vs. 224):** la ablación sugiere que 160 px cuesta ≈0.07 de AUC; si el objetivo es el borde, hay que justificar ese compromiso con cifras o adoptar 224 px (2× el cómputo), después de verificar A13 (7.1, punto 3).
* **Redacción del trabajo:** el título y el planteamiento destacan el modelado temporal con Transformers; estos resultados no muestran que el módulo temporal ni la atención aporten frente a un promedio simple o a un LSTM. Conviene reportarlo tal cual y plantear qué se concluye de ello (sección 5.2, punto 2).

## 8. Reproducción y archivos

* Configuración: `config/ablation_modelling.yaml`, `config/a0_convnextv2_femto.yaml`; diseño en `ablation_definition.md`; implementación y decisiones en `README.md`.
* Corridas: `outputs/modelling/{id}_s{semilla}/` (`best.pt`, `run.yaml`, `history.csv`, `metrics_val.json`, `metrics_test.json`, `preds_*.npz`, `variant.json`) y un reporte por corrida en `reports/*_ablation_run_{id}_s{semilla}.md/.json` (parámetros completos, entorno, métricas y comando para repetirla).
* Tablas: `results/ablation_modelling.md/.csv`, `results/scoring_M1-M12.md`; copias con fecha en `results/history/`.
* Para regenerar: `python 2_report.py` (el bootstrap con IC por ventana tarda varios minutos) y `python 3_scoring_ablation.py`.
