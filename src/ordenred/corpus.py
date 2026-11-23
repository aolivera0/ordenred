"""Lectura, escritura y validación del corpus (contrato C5). Dueño: rol 5.

Todos los módulos cargan datos con `cargar_jsonl`; el generador valida cada registro con
`validar_registro` antes de escribirlo.
"""
from __future__ import annotations

import json
from pathlib import Path

from .arbol import desde_corchetes, hojas, preterminales
from .semantica import ACCIONES, FormaLogica

OBLIGATORIOS = ("id", "orden", "intencion", "tokens", "etiquetas", "arbol",
                "formas_logicas", "iptables", "acl", "ambigua", "fuente")
FUENTES = ("sintetico", "humano")


def cargar_jsonl(ruta: str | Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def guardar_jsonl(registros: list[dict], ruta: str | Path) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validar_registro(r: dict) -> list[str]:
    """Lista de errores del registro (vacía si cumple C5).

    En registros humanos fuera de la cobertura de la gramática, `arbol` y `etiquetas` pueden ser null.
    """
    errores = [f"falta el campo {c!r}" for c in OBLIGATORIOS if c not in r]
    if errores:
        return errores
    if r["fuente"] not in FUENTES:
        errores.append(f"fuente inválida: {r['fuente']!r}")
    if r["intencion"] not in ACCIONES:
        errores.append(f"intención inválida: {r['intencion']!r}")
    if not isinstance(r["tokens"], list) or not all(isinstance(t, str) for t in r["tokens"]):
        errores.append("tokens debe ser una lista de cadenas")
    if not isinstance(r["ambigua"], bool):
        errores.append("ambigua debe ser booleano")
    if r.get("orden_ruidosa") is not None and not isinstance(r["orden_ruidosa"], str):
        errores.append("orden_ruidosa debe ser cadena o null")
    for campo in ("iptables", "acl"):
        if not isinstance(r[campo], list) or not all(isinstance(x, str) for x in r[campo]):
            errores.append(f"{campo} debe ser una lista de cadenas")

    sin_arbol = r["arbol"] is None or r["etiquetas"] is None
    if sin_arbol and r["fuente"] != "humano":
        errores.append("solo los registros humanos pueden tener arbol/etiquetas en null")
    if not sin_arbol:
        if len(r["etiquetas"]) != len(r["tokens"]):
            errores.append("etiquetas y tokens tienen longitudes distintas")
        try:
            a = desde_corchetes(r["arbol"])
            if [t.forma for t in hojas(a)] != r["tokens"]:
                errores.append("las hojas del árbol no coinciden con tokens")
            if preterminales(a) != r["etiquetas"]:
                errores.append("los preterminales del árbol no coinciden con etiquetas")
            if a.etiqueta != "S":
                errores.append(f"la raíz del árbol es {a.etiqueta!r}, no 'S'")
        except ValueError as e:
            errores.append(f"árbol mal formado: {e}")

    if not isinstance(r["formas_logicas"], list) or not r["formas_logicas"]:
        errores.append("formas_logicas debe ser una lista no vacía")
    else:
        for i, d in enumerate(r["formas_logicas"]):
            try:
                fl = FormaLogica.desde_dict(d)
                if fl.accion != r["intencion"]:
                    errores.append(f"formas_logicas[{i}].accion no coincide con intencion")
            except (TypeError, ValueError) as e:
                errores.append(f"formas_logicas[{i}] inválida: {e}")
    return errores
