"""Persistencia en JSON de los modelos del rol 4 (acuerdo C6). Dueño: rol 4.

Cada modelo se guarda como un objeto JSON UTF-8 con "tipo" (nombre de la clase) y "version"; lo usan
`NaiveBayes`, `RegresionLogistica` y `HMM`. Nunca se escriben Infinity ni NaN.
"""
from __future__ import annotations

import json
from pathlib import Path

VERSION = 1


def escribir_json(ruta: str | Path, datos: dict) -> None:
    """Escribe `datos` en `ruta` (crea las carpetas que falten)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: el JSON nunca lleva Infinity ni NaN (falla en vez de escribir algo ilegible).
    texto = json.dumps(datos, ensure_ascii=False, indent=1, allow_nan=False)
    ruta.write_text(texto + "\n", encoding="utf-8")


def leer_json(ruta: str | Path, tipo: str, campos: tuple[str, ...] = ()) -> dict:
    """Lee un modelo guardado con `escribir_json`.

    ValueError si el contenido no es un objeto JSON, si "tipo" no coincide, si la versión no es
    `VERSION` o si falta alguno de `campos`.
    """
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    if not isinstance(datos, dict) or datos.get("tipo") != tipo:
        recibido = datos.get("tipo") if isinstance(datos, dict) else None
        raise ValueError(f"el archivo no es un modelo {tipo} (tipo={recibido!r})")
    if datos.get("version") != VERSION:
        raise ValueError(f"versión no soportada: {datos.get('version')!r}; se esperaba {VERSION}")
    faltan = [c for c in campos if c not in datos]
    if faltan:
        raise ValueError(f"al modelo {tipo} le faltan los campos {faltan}")
    return datos
