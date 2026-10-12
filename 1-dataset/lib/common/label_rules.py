"""
Reglas de palabras clave que dan a un item descargado una categoría
provisional a partir de su título/descripción (no se mira el video). Las usa
2_filter.py para decidir si un item descubierto es lo bastante relevante como
para descargarlo, y para enfocar una corrida en clases concretas (--classes).

Las categorías son las 13 categorías de anomalía de UCF-Crime (Normal es el
respaldo por palabras de cámara de más abajo, no una regla de palabras clave);
classes.py las reduce a las clases con las que se entrenan los modelos.
"""

# ── Etiqueta provisional a nivel de item (a partir de metadatos, no de un VLM) ─
#
# Cubre las 13 categorías de anomalía de UCF-Crime (Normal es el respaldo por
# palabras de cámara de más abajo, no una regla de palabras clave). Se evalúa
# de arriba hacia abajo; gana la primera coincidencia. El título se revisa
# antes que la descripción. Las palabras clave no distinguen mayúsculas y
# cuentan las coincidencias parciales.
#
# El orden importa: las categorías con palabras amplias (p. ej. "robo" suelto
# en Robbery) van DESPUÉS de las que tienen frases compuestas más específicas
# que, si no, quedarían tapadas (p. ej. "robo a casa" de Burglary, "robo en
# tienda" de Shoplifting); ver el grupo de hurtos (Burglary, Shoplifting,
# Robbery, Stealing) más abajo.
#
# Las etiquetas son directamente nombres de categoría de UCF-Crime, así que
# se escriben tal cual en la columna ucf_category de la tabla items, sin otra
# traducción. "description" es una frase corta legible para cada categoría (la
# usaba solo el prompt del VLM del pipeline asistido por IA, que no forma
# parte de este workspace).

LABEL_RULES = [
    {
        "label": "Shooting",
        "description": "shooting or gunfire",
        "keywords": [
            # Español
            "balacera", "tiroteo", "disparo", "disparos", "balazo",
            "baleado", "baleada", "rafagueado", "fusilamiento",
            "sicario", "sicariato", "asesinado a tiros",
            "arma de fuego", "pistola", "rifle",
            # Portugués
            "tiroteio", "fuzilamento", "tiros",
            # Inglés
            "shooting", "gunshot", "gunfire", "shot dead",
        ],
    },
    {
        "label": "Explosion",
        "description": "explosion or bomb blast",
        "keywords": [
            # Español
            "explosión", "explosion", "estallido", "bomba",
            "detonación", "detonacion", "explota",
            # Portugués
            "explosão",
            # Inglés
            "explosion", "bomb", "blast", "detonation",
        ],
    },
    {
        "label": "Arson",
        "description": "arson / deliberate fire-setting",
        "keywords": [
            # Español
            "incendio provocado", "incendio intencional", "prenden fuego",
            "queman", "pirómano", "piromano",
            # Portugués
            "incêndio criminoso", "atearam fogo", "incendiário",
            # Inglés
            "arson", "set fire", "set on fire", "deliberate fire",
        ],
    },
    {
        "label": "Burglary",
        "description": "burglary / breaking into a home or building to steal",
        "keywords": [
            # Español
            "robo a casa", "robo a vivienda", "robo a domicilio", "allanamiento",
            "violación de domicilio", "entran a robar", "entraron a robar",
            # Portugués
            "arrombamento", "invasão de residência", "furto a residência",
            # Inglés
            "burglary", "break-in", "break in", "home invasion", "broke into",
        ],
    },
    {
        "label": "Shoplifting",
        "description": "shoplifting / theft from a store",
        "keywords": [
            # Español
            "hurto en tienda", "robo en tienda", "hurto de mercancía",
            "hurto de mercaderia", "robo en supermercado", "hurto en supermercado",
            # Portugués
            "furto em loja", "furto em supermercado",
            # Inglés
            "shoplifting", "shoplifter",
        ],
    },
    {
        "label": "Robbery",
        "description": "robbery / theft with force or a weapon",
        "keywords": [
            # Español — genéricas
            "robo", "roban", "asalto", "asaltan", "asaltaron", "asaltante",
            "ladrón", "ladron", "ladrones", "delincuente", "hampón", "hampones",
            # Español — modalidades
            "cogoteo", "arrebato", "arrebatan",
            "portonazo", "fleteo", "raponazo",
            "salidera", "entradera", "robo express",
            "jalón de bolsa", "jalón", "cosquilleo",
            "pirañas",
            "robo a mano armada", "asalto a mano armada",
            "robo de celular", "robo de moto", "robo de auto", "robo de carro",
            "carterista", "bolsista",
            "secuestro", "secuestran", "secuestrado", "rapto", "levantón", "plagio",
            # Portugués
            "assalto", "roubo", "roubaram", "ladrão", "ladrao",
            "arrastão", "sequestro relâmpago", "sequestro",
            # Inglés
            "robbery", "mugging", "carjacking", "pickpocket",
            "kidnap", "kidnapping", "hold up",
        ],
    },
    {
        "label": "Stealing",
        "description": "stealing / theft without violence or confrontation",
        "keywords": [
            # Español
            "hurto", "hurtan", "hurtaron", "sustracción", "sustraccion",
            "hurto de bicicleta", "hurto de paquete", "hurto simple",
            # Portugués
            "furto", "furtaram", "furto simples",
            # Inglés
            "stealing", "theft", "package theft", "bike theft", "stole",
        ],
    },
    {
        "label": "Abuse",
        "description": "abuse (domestic, child, elder, or animal)",
        "keywords": [
            # Español
            "abuso infantil", "abuso a menor", "maltrato infantil", "maltrato animal",
            "violencia doméstica", "violencia intrafamiliar", "abuso sexual",
            "maltrato a anciano", "violencia de género",
            # Portugués
            "abuso infantil", "maus-tratos", "violência doméstica",
            # Inglés
            "child abuse", "elder abuse", "animal abuse", "domestic abuse", "domestic violence",
        ],
    },
    {
        "label": "Assault",
        "description": "assault / physical attack on a person",
        "keywords": [
            # Español
            "agresión", "agresion", "golpiza", "paliza", "puñetazo",
            "apuñalamiento", "cuchillazo", "navajazo", "acuchillado", "acuchillada",
            "golpeado", "golpeada", "ataque a", "atacan a", "atacaron a",
            # Portugués
            "agressão", "agressao", "espancamento", "esfaqueado", "surra",
            # Inglés
            "assault", "stabbing", "stabbed", "beaten", "attacked",
        ],
    },
    {
        "label": "Fighting",
        "description": "fighting / mutual brawl",
        "keywords": [
            # Español
            "pelea", "riña", "rina", "gresca", "bronca", "trifulca",
            "batalla campal", "pleito",
            # Portugués
            "briga", "pancadaria",
            # Inglés
            "fight", "brawl", "fistfight",
        ],
    },
    {
        "label": "Arrest",
        "description": "police arrest / detention",
        "keywords": [
            # Español
            "detención", "detencion", "arresto", "captura policial",
            "atrapan a", "detienen a", "operativo policial", "cae detenido", "cae preso",
            # Portugués
            "detenção", "prisão", "prendem",
            # Inglés
            "arrest", "arrested", "police arrest", "detained by police", "taken into custody",
        ],
    },
    {
        "label": "Vandalism",
        "description": "vandalism / property damage",
        "keywords": [
            # Español
            "vandalismo", "destrozos", "destrozan", "grafiti", "pintadas",
            "daños a la propiedad", "danos a la propiedad",
            "rompen vidrios", "rompieron vidrios",
            # Portugués
            "vandalizaram", "pichação",
            # Inglés
            "vandalism", "graffiti", "property damage", "vandalized",
        ],
    },
    {
        "label": "Road_Accident",
        "description": "road accident / vehicle collision",
        "keywords": [
            # Español
            "atropello", "atropelló", "atropellado", "atropellada", "atropellan",
            "accidente de tránsito", "accidente vial", "accidente de tráfico",
            "accidente de carro", "accidente de moto",
            "choque", "colisión", "colision",
            "volcamiento", "volcó", "volco", "vuelco", "se volcó",
            # Portugués
            "atropelamento", "atropelado",
            "acidente de trânsito", "acidente de carro",
            "capotamento", "engavetamento", "batida",
            # Inglés
            "road accident", "car accident", "traffic accident",
            "hit and run", "run over", "crash",
        ],
    },
    # "Normal" NO se asigna por palabras clave aquí: es el respaldo por palabras de cámara de abajo.
]

# Palabras de cámara/vigilancia: si aparece alguna (sin coincidencia de delito)
# el video se considera Normal.
CAMERA_KEYWORDS = [
    "cámara", "camara", "câmera",
    "circuito cerrado", "circuito fechado",
    "cctv",
    "videovigilancia", "vigilância", "vigilancia",
    "cámara de seguridad", "camara de seguridad",
    "footage", "surveillance",
    "serenazgo", "grabado", "grabó", "captó", "captado", "registró",
    "flagrou", "filmou",
]


def assign_label(title: str, description: str) -> str | None:
    """
    Devuelve la categoría UCF-Crime provisional del item (una de las 13
    nombradas en LABEL_RULES, o 'Normal'), o None si nada coincide. En todas
    las reglas de delito se revisa el título antes que la descripción (la
    señal más fiable gana); el respaldo 'Normal' solo mira el título.
    """
    title_lc = title.lower()
    desc_lc  = description.lower()

    for rule in LABEL_RULES:
        kws = [k.lower() for k in rule["keywords"]]
        if any(kw in title_lc for kw in kws):
            return rule["label"]

    for rule in LABEL_RULES:
        kws = [k.lower() for k in rule["keywords"]]
        if any(kw in desc_lc for kw in kws):
            return rule["label"]

    if any(kw.lower() in title_lc for kw in CAMERA_KEYWORDS):
        return "Normal"

    return None
