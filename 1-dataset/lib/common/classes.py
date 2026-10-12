"""
Vocabulario de clases de la etapa de dataset.

RAW_TO_TARGET reduce las categorías finas estilo UCF-Crime que pueden asignar
las reglas de palabras clave (label_rules.py) a las clases con las que se
entrenan los modelos: Normal, Robbery, Assault, Road_Accident, Shooting, más
Others.

`Others` NO es una clase de entrenamiento. Es un cajón para material real y
utilizable que no encaja en ninguna de las cinco clases (vandalismo, incendio
provocado, explosiones, arrestos, hurto en tienda, maltrato). Se conserva en
disco durante la edición manual y se excluye de la división train/val/test
(ver lib/splitting/).

La asignación es un criterio tomado para esta tesis, no una taxonomía
estándar. Vandalism/Arson/Explosion se plegaban originalmente en
Assault/Shooting; una corrida de prueba mostró Assault con 0 % de recall y
precisión, atribuido a que Vandalism (daño a propiedad, sin personas) y Arson
(fuego/humo) no comparten ninguna firma visual con Fighting, y a que
Explosion no se parece en nada a un disparo; por eso pasaron a Others.
"""

RAW_TO_TARGET = {
    'Normal':        'Normal',
    'Robbery':       'Robbery',
    'Burglary':      'Robbery',
    'Shoplifting':   'Others',
    'Stealing':      'Robbery',
    'Assault':       'Assault',
    'Fighting':      'Assault',
    'Abuse':         'Others',
    'Arrest':        'Others',
    'Vandalism':     'Others',
    'Arson':         'Others',
    'Road_Accident': 'Road_Accident',
    'Shooting':      'Shooting',
    'Explosion':     'Others',
}


def to_target_class(raw_class):
    """Convierte la clase cruda de un item/clip en su clase de entrenamiento.

    Lanza KeyError si `raw_class` no está en RAW_TO_TARGET: agrégala ahí en
    lugar de descartar datos o clasificarlos mal en silencio.
    """
    try:
        return RAW_TO_TARGET[raw_class]
    except KeyError:
        raise KeyError(
            f'No label mapping for class {raw_class!r} — add it to '
            f'RAW_TO_TARGET in utils/labels.py.'
        ) from None


TARGET_CLASSES = sorted(set(RAW_TO_TARGET.values()))


def target_class_of(raw_label):
    """Convierte una etiqueta cruda de item/clip (una categoría de LABEL_RULES
    o 'Normal') en su clase destino, o None si no hay etiqueta: un item sin
    etiquetar nunca debe cumplir un filtro --classes."""
    return to_target_class(raw_label) if raw_label else None


def parse_classes(raw):
    """Convierte el valor de --classes (nombres de clase separados por comas,
    p. ej. 'Shooting,Assault,Road_Accident') en un conjunto validado, o None
    si está vacío (sin filtro: todo pasa).

    Lanza ValueError (indicando las opciones válidas) si algún nombre no es
    una clase conocida; quien llame debe capturarlo y reportarlo con
    parser.error() para dar un error de CLI limpio en vez de un traceback.
    """
    if not raw:
        return None
    classes = {c.strip() for c in raw.split(",") if c.strip()}
    unknown = classes - set(TARGET_CLASSES)
    if unknown:
        raise ValueError(
            f"unknown class(es) {sorted(unknown)} — valid choices are {TARGET_CLASSES}"
        )
    return classes
