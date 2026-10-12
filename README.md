# Workspace de la tesis — paquete para revisión por pares

Código de un clasificador de incidentes delictivos sobre material de
Latinoamérica, organizado en tres etapas que se revisan en orden. Cada carpeta de
etapa tiene su propio README con la instalación, los comandos, las notas de diseño
y lo que se verificó.

| Etapa | Carpeta | Estado |
|---|---|---|
| 1. Dataset: scraping, edición manual, división y validación humana | [`1-dataset/`](1-dataset/README.md) | listo para revisión (pipeline de 8 pasos, estado en CSV) |
| 2. Modelado: ConvNeXtV2 + transformer temporal, puntuación de video con MIL | [`2-modelling/`](2-modelling/README.md) | listo para revisión (entrenar y evaluar; métricas de referencia reproducidas) |
| 3. Optimización: destilación, poda y cuantización | `3-optimization/` | por migrar |

El modelo final es el estudiante MobileViT-XS destilado
(`2.5-model/models/mil_opt/distil_convbase_v2.py` en el árbol de trabajo); las
etapas 2 y 3 contendrán únicamente los archivos de los que depende ese script.

Convención: en cada etapa, los scripts principales están en la raíz de la carpeta
con prefijo numérico según el orden del pipeline, y la lógica y las utilidades de
cada parte viven en `lib/`.
