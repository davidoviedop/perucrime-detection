# Etapa 1 — Dataset

Cómo se construye el dataset de material de incidentes de Latinoamérica
(`perucrime-5`): desde el scraping de videos candidatos hasta una división
train/val/test sin fugas. Cada paso es un script de línea de comandos o un sitio web
pequeño. El estado del pipeline vive en archivos CSV
legibles (`data/metadata/`), como los metadatos que acompañan a los datasets
públicos, y los videos del dataset se *generan* a partir de esos CSV.


## Idea central: el CSV manda, los videos son derivados

```
 1_discover ─► 2_filter ─► 3_download          4_edit ─► 5_render ─► 6_validate ─► 7_split ─► 8_export
 URLs          etiqueta    videos crudos        decidir    generar     clase /      split por    carpeta
 (items.csv)   y filtro    data/1_raw/          clips      los videos  descartar    ventanas     final
                                                (clips.csv)(2_clips/)  (clips.csv)  (split.csv)  (3_dataset/)
```

* En `4_edit` no se codifica ningún video: cada clip es una **fila de `clips.csv`**
  (de qué video crudo sale, ventana `start_sec`–`end_sec`, caja de crop, clase).
* `5_render` corta esos videos del crudo, una sola vez por clip, y anota en el CSV
  las medidas del resultado (fotogramas, fps, tamaño, duración).
* `6_validate` solo permite **asignar la clase o descartar**; todo queda en `clips.csv`.
* `7_split` asigna un split a cada item (`split.csv`) y `8_export` arma la carpeta
  final `{fuente}/{split}/{clase}/` que lee el cargador de datos del Modelado.

Como la edición solo cambia filas, es reversible y reproducible: no se recodifica un
clip ya recodificado, y toda la carpeta final se puede borrar y regenerar desde los CSV.

## Estructura y numeración

Los scripts de la raíz están numerados en el orden del pipeline; la lógica y las
utilidades de cada parte viven en `lib/`.

```
1-dataset/
├── 1_discover.py   Paso 1  descubrir URLs de video (YouTube, Facebook, TikTok…)  → items.csv
├── 2_filter.py     Paso 2  metadatos + etiqueta por palabras clave + filtro      → items.csv
├── 3_download.py   Paso 3  descargar los videos aprobados                        → data/1_raw/{país}/{item_id}.mp4
├── 4_edit.py       Paso 4  sitio web: decidir los clips sobre los videos crudos  → clips.csv
├── 5_render.py     Paso 5  generar los videos de los clips desde el crudo        → data/2_clips/{clip_id}.mp4
├── 6_validate.py   Paso 6  sitio web: validación humana (clase o descartar)      → clips.csv
├── 7_split.py      Paso 7  división train/val/test balanceada por ventanas       → split.csv
├── 8_export.py     Paso 8  exportar el dataset final                             → data/3_dataset/{fuente}/{split}/{clase}/ + manifest.csv
│
├── config/         sources.yaml (qué descubrir), cookies/
├── data/           (git-ignorado, salvo metadata/) metadata/  1_raw/  2_clips/  external/  3_dataset/  logs/
├── requirements.txt
└── lib/            lógica y utilidades, una carpeta por parte
    ├── common/     clases, reglas de palabras clave, almacén CSV (csvstore), clips.csv (clips.py), esquemas, rutas, logger
    ├── scraping/   pasos 1-3: almacén de items, ajustes de CLI, descubrimiento, filtrado, descarga
    ├── editing/    paso 4: la app Flask de edición
    ├── rendering/  paso 5: corte con ffmpeg y medición de los videos generados
    ├── validation/ paso 6: la app Flask de validación
    ├── splitting/  paso 7: pool de items, búsqueda de asignación, split.csv
    ├── exporting/  paso 8: armado de la carpeta final, manifest.csv, verificación
    └── webkit/     código compartido por los dos sitios (ffmpeg, streaming, validación, CSS/JS)
```

> Las carpetas de `lib/` no llevan número porque son paquetes de Python y un
> identificador de Python no puede empezar con un dígito. El orden del pipeline lo
> marcan los scripts de la raíz y las carpetas de `data/`.

## Los CSV (`data/metadata/`)

UTF-8, con cabecera, una fila por registro. Las columnas están documentadas en
`lib/common/schemas.py`.

| Archivo | Una fila por… | Lo escriben | Contenido principal |
|---|---|---|---|
| `items.csv` | video fuente descubierto | pasos 1-3 | `id` (= `sha256(url)[:16]`), `url`, `source`, `country`, título, descripción, `duration`, `view_count`, `provisional_label`, `filter_reason`, `status`, `download_path` |
| `clips.csv` | clip decidido | pasos 4, 5, 6 | `clip_id`, `item_id`, `raw_file`, `class`, `label`, `start_sec`, `end_sec`, `crop_*`, `status` (`pending`/`rendered`), `discarded`, `n_frames`, `fps`, `width`, `height`, `duration`, `source_url`, `title` |
| `split.csv` | item (fuente + `item_id`) | paso 7 | `source`, `item_id`, `split`, `n_clips`, `windows` |
| `3_dataset/manifest.csv` | video exportado | paso 8 | `item_id`, `clip_id`, `source`, `class`, `label`, `split`, `filename`, `output_path`, `start_sec`, `end_sec`, `source_url` |

Cómo se guardan: escritura atómica (archivo temporal + `os.replace`) y bloqueo entre
procesos (`fcntl.flock`), de modo que se pueden dejar abiertos los sitios web mientras
corre `5_render.py` sin corromper nada. Los scripts de scraping abren `items.csv` en
modo exclusivo (agrupan los guardados cada 25 cambios; si se mata el proceso a la
fuerza se pierden, como máximo, los últimos 25). Detalles en `lib/common/csvstore.py`.

## Instalación

```bash
pip install -r requirements.txt
# sistema: ffmpeg/ffprobe en el PATH; Deno instalado (las descargas de YouTube lo necesitan)
```

* Todos los comandos se ejecutan desde esta carpeta (`1-dataset/`).
* `config/cookies/*` está en `.gitignore`. Las cookies (`cookies.txt` en formato
  Netscape) solo hacen falta para videos de YouTube con restricción de edad y para
  Facebook/Instagram.

## Cómo ejecutar cada paso

### Pasos 1-3 — Scraping

```bash
python 1_discover.py --source youtube --countries PE,CO     # URLs → items.csv (estado: discovered)
python 2_filter.py                                          # metadatos + etiqueta → approved | rejected
python 3_download.py                                        # aprobados → data/1_raw/{país}/{item_id}.mp4
```

* Las fuentes de descubrimiento (consultas, canales, páginas y cuentas por país) están
  en `config/sources.yaml`.
* `2_filter.py` rechaza por duración (8–180 s por defecto), vistas, palabras de rechazo
  en el título (tráilers, videoclips, …) y "sin etiqueta". La etiqueta sale de las
  reglas de palabras clave de `lib/common/label_rules.py` (el título se revisa antes
  que la descripción).
* `--classes Shooting,Assault,Road_Accident` enfoca una corrida en las clases escasas:
  preselecciona las consultas de YouTube en `1_discover.py` y filtra por etiqueta en
  `2_filter.py` y `3_download.py`.
* Todo script registra en `data/logs/` (detalle completo en DEBUG: es el rastro de auditoría).
* `2_filter.py --stats` muestra el conteo de etiquetas y motivos de rechazo; `--reset`
  vuelve a encolar los rechazados.

### Paso 4 — Edición: decidir los clips

```bash
python 4_edit.py            # abrir http://localhost:5050
```

Vista *Raw*: se elige un video crudo, se ajusta la ventana de recorte y una caja de
crop opcional sobre el propio video, se escoge la clase y **Guardar clip** agrega una
fila a `clips.csv`. De un mismo video se pueden registrar varios clips (el incidente y
un tramo tranquilo como `Normal`). Vista *Clips*: solo los videos con clips, filtrables
por clase. En ambas, un clip registrado se puede reclasificar, cargar en el editor para
cambiar su ventana/crop, o eliminar, y se pueden reclasificar de una vez los clips
anómalos de un video. El video crudo nunca se modifica.

### Paso 5 — Generar los videos

```bash
python 5_render.py              # genera lo que falte (incremental)
python 5_render.py --dry-run    # solo lista qué se generaría
python 5_render.py --force      # regenera todo; --prune borra videos huérfanos
```

Corta cada clip del video crudo con los parámetros del CSV (una sola recodificación:
recorte + crop) y anota fotogramas/fps/tamaño/duración. Solo regenera los clips
pendientes, los que cambiaron de ventana o crop y los que perdieron su archivo;
cambiar solo la clase no obliga a regenerar. Los descartados se omiten.

### Paso 6 — Validación humana

```bash
python 6_validate.py        # abrir http://localhost:5051
```

Se revisan los videos generados y a cada clip se le **asigna la clase correcta o se
descarta** (también por item, o en bloque para todos los items listados). Descartar
no borra nada: marca `discarded = 1` y se puede restaurar. La ventana y el crop no se
tocan aquí; si hay que cambiarlos se vuelve al paso 4 y se ejecuta de nuevo el 5.
Esta etapa cubre los clips de `scrapped` (los de `clips.csv`).

### Paso 7 — División

```bash
python 7_split.py --stats        # solo reporte del pool
python 7_split.py --dry-run      # calcula y muestra, sin escribir
python 7_split.py                # escribe data/metadata/split.csv
```

Balancea **ventanas de entrenamiento**, no clips, en 70/15/15 por clase (ver
[Notas de diseño](#notas-de-diseño)). Para `scrapped` usa los clips activos de
`clips.csv`; ucf-crime y vioperu se leen de `data/external/{fuente}/{clase}/*.mp4`.
Si quedan clips activos sin generar, aborta y pide ejecutar `5_render.py`.

### Paso 8 — Exportar el dataset final

```bash
python 8_export.py                   # copia los videos
python 8_export.py --mode hardlink   # enlaces duros, sin duplicar espacio (mismo disco)
```

Arma `data/3_dataset/{fuente}/{split}/{clase}/` y `manifest.csv`, y verifica que estén
todos los archivos y que ningún item aparezca en más de un split. Al repetirlo borra los
archivos de la exportación anterior que ya no corresponden. Si la validación o la edición
cambian algo, el orden es: `5_render` → `7_split` → `8_export`.

## Notas de diseño

* **Clases.** Las cinco clases de entrenamiento son `Normal, Robbery, Assault,
  Road_Accident, Shooting`. `Others` es un cajón para material utilizable que no
  encaja en ninguna (vandalismo, incendio provocado, explosiones, arrestos, hurto en
  tienda, maltrato); nunca se divide ni se exporta. Las reglas de palabras clave
  producen las 13 categorías finas de UCF-Crime y `lib/common/classes.py:RAW_TO_TARGET`
  las reduce.
* **Por qué la división cuenta ventanas.** Los modelos se entrenan con ventanas
  deslizantes (16 fotogramas, paso de fotogramas 4, paso de ventana 16, ≤70 ventanas
  por clip). La duración de los clips varía mucho entre fuentes, así que una división
  70/15/15 por clips queda lejos de 70/15/15 en ventanas. `lib/splitting/` puntúa la
  división por cantidad de ventanas **y** de items (un término de diversidad, porque
  unos pocos items "ballena" pueden acumular casi todas las ventanas de una clase), con
  búsqueda voraz + escalada + intercambio de pares desde varias semillas. Las constantes
  de ventana de `lib/splitting/config.py` deben coincidir con las de la etapa de Modelado.
* **Sin fugas.** La unidad de asignación es el *item* (todos los clips de un video
  fuente, o un archivo en `ucf-crime`/`vioperu`), nunca el clip.
* **Por qué una huella (`render_key`).** Cada fila guarda un hash de los parámetros con
  los que se generó su video. Si alguien edita el CSV a mano, o el video se borra, el
  paso 5 lo detecta y lo regenera; si el clip se edita mientras se está generando, no se
  marca como al día.
* **Limitación conocida: concentración en Robbery.** Al hacer la división final, Robbery
  tenía pocos items y un puñado de ellos acumulaba la mayoría de sus ventanas, de modo
  que incluso la mejor división dejaba su val/test dominado por unos pocos items (la tabla
  de diversidad que imprime `7_split.py --dry-run` muestra las cifras actuales). La
  solución es conseguir más material de Robbery, más corto y más diverso; volver a dividir
  no la resuelve.

## Fuentes externas

`ucf-crime` y `vioperu` son datasets públicos agregados a mano; ningún código de este
paquete los produce y **no están en `clips.csv`**. Se colocan en
`data/external/{fuente}/{clase}/{archivo}.mp4` y las herramientas solo asumen que **un
archivo es un video fuente independiente** (su nombre sin extensión es su id de item).
`7_split.py` y `8_export.py` los tratan igual que a `scrapped`.

> **Pendiente de completar por la autora/el autor antes de la revisión:** cómo se
> eligieron los subconjuntos y cómo se reetiquetaron a las cinco clases, las citas y
> licencias de los datasets originales y la cantidad de clips tomados de cada uno.

## Alcance: qué incluye y qué no

**Incluye:** scraping (descubrir/filtrar/descargar), el sitio de edición, la generación
de los clips, la validación humana, la división balanceada por ventanas y la exportación.

**Se excluye a propósito:**
* Los agentes asistidos por IA (detección de bordes y clasificación de clips con un VLM)
  que se usaron en el trabajo original solo para acelerar la anotación. Quedan fuera de
  este paquete: la vía de revisión son las herramientas humanas de arriba (`4_edit.py`,
  `6_validate.py`).
* El pipeline de procesamiento con base de datos (`1.7-dataset`: extracción por detección
  de escenas, render, interfaz de verificación con BD) y `3-migrate.py`, que solo hacía
  de puente de la BD de staging hacia ese pipeline.

## Qué cambió respecto de los scripts originales

Este paquete **no es una copia 1:1**: se rediseñó el manejo del estado (sin MySQL, con
CSV) y la edición (decisiones registradas y videos generados después). Equivalencias:

| Original | Aquí |
|---|---|
| `0-scrapping/0-scrapping.py` | `1_discover.py` + `lib/scraping/discovery/` |
| `0-scrapping/1-filtering.py` | `2_filter.py` + `lib/scraping/filtering/` |
| `0-scrapping/2-download.py` | `3_download.py` + `lib/scraping/downloader.py` |
| tabla `items` de MySQL (`utils/db.py`) | `data/metadata/items.csv` (`lib/scraping/store.py` sobre `lib/common/csvstore.py`) |
| `1-processing/postediting.py` (recodificaba al instante) | `4_edit.py` (registra filas en `clips.csv`) + `5_render.py` (genera los videos) |
| `1.7-dataset/8-postediting.py` (trim/crop/clase sobre archivos) | `6_validate.py` (solo clase o descartar, en `clips.csv`) |
| `perucrime-5/window_resplit.py` (movía archivos) | `7_split.py` (escribe `split.csv`) + `8_export.py` (arma la carpeta final) |
| `utils/labels.py` | `lib/common/classes.py` + `lib/common/label_rules.py` |

Otras diferencias deliberadas: una sola lista de clases (con `Others`) reemplaza dos
copias ligeramente distintas; el orden de las ventanas del algoritmo de asignación ya
no depende del orden de los archivos/filas (se ordena por id), por lo que el resultado
es reproducible; se quitaron las credenciales y la dependencia de MySQL; el CSS y JS de
las plantillas HTML están en archivos estáticos; los comentarios, docstrings, textos de
`--help` y este README están en español. Los mensajes que se imprimen al ejecutar (logs)
y los mensajes de error de la API se dejaron en inglés, como en el original.

## Cómo se comprobó

Con datos sintéticos (videos reales de ffmpeg, un navegador real y yt-dlp simulado):

* **Capa CSV** (`csvstore`): comillas, comas, saltos de línea y unicode; dos instancias
  que ven los cambios de la otra; una transacción fallida no deja residuos; el modo
  exclusivo rechaza a un segundo proceso y agrupa los guardados; escrituras desde 6
  hilos sin pérdidas.
* **Scraping** de punta a punta con `items.csv` real (1→2→3, `--classes`, `--stats`,
  `--reset`, repetir sin duplicar) y, contra el código original, mismas llamadas a
  yt-dlp, mismos logs y `evaluate()` idéntico en 1215 combinaciones.
* **Edición** (API y navegador real): guardar, editar ventana/crop, reclasificar, borrar,
  vista Clips y sus filtros, errores de validación, persistencia entre instancias.
* **Generación**: dimensiones del crop, cantidad de fotogramas, corrida incremental,
  cambiar solo la clase (no regenera), cambiar la ventana (regenera), descartados, archivo
  borrado a mano, huérfanos, video crudo ausente, `--force`.
* **Validación** (API y navegador real): clase, descartar/restaurar, filtros de estado,
  clip sin generar, reclasificación en bloque, y que rechace cambios de ventana/crop.
* **División y exportación**, contra el script original sobre el mismo pool: mismas
  ventanas por item y clase, **misma asignación de splits**, mismos archivos exportados;
  además descartados/`Others` excluidos, aborto si hay clips sin generar, limpieza de
  obsoletos, modos `copy`/`hardlink`/`symlink` y comprobación de fugas en disco.
