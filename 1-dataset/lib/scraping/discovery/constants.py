"""Tablas de plataformas / países y constantes compartidas por los módulos de descubrimiento."""

COUNTRY_KEY_MAP = {
    "peru":      "PE",
    "brazil":    "BR",
    "colombia":  "CO",
    "argentina": "AR",
    "chile":     "CL",
    "ecuador":   "EC",
    "mexico":    "MX",
    "venezuela": "VE",
}

# Plataforma → clave de sources.yaml que contiene la lista de URLs a raspar
PLATFORM_LIST_KEY = {
    "facebook":  "pages",
    "tiktok":    "accounts",
    "instagram": "accounts",
    "twitter":   "accounts",
}

SUPPORTED_SOURCES = ["youtube", "facebook", "tiktok", "instagram", "twitter"]


# Errores de extracción consecutivos (no "0 resultados": una excepción real de
# yt-dlp) tras los cuales un bucle de descubrimiento se corta antes. Unas
# pocas URLs malas son normales (cuentas renombradas o borradas); tantas
# seguidas en cuentas independientes indican que el extractor mismo está caído
# (p. ej. el extractor instagram:user de yt-dlp marcado como roto; ver su
# advertencia "marked as broken") y no que la lista de fuentes esté obsoleta;
# seguir con el resto de la lista solo repetiría el mismo fallo.
MAX_CONSECUTIVE_ERRORS = 3
