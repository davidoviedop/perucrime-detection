"""
Código compartido por los dos sitios web (4_edit.py y 6_validate.py): wrappers de
ffmpeg, caché de duraciones, streaming de video con peticiones Range, validación de
peticiones y los recursos estáticos (CSS/JS) que cargan ambos front-ends.

Ningún sitio usa base de datos: su estado vive en data/metadata/clips.csv (ver
lib/common/clips.py) y los videos se leen de disco bajo demanda.
"""
