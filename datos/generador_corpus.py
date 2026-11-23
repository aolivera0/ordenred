"""Generador del corpus sintético y del treebank. Dueño: rol 5.

Estrategia (no circular): se muestrea primero una FormaLogica y luego se realiza en texto
construyendo al mismo tiempo el árbol de oro con la gramática. Las reglas de oro salen de
`a_iptables` / `a_acl_cisco` aplicadas a la forma lógica de oro.

Uso previsto:
    PYTHONPATH=src python datos/generador_corpus.py --n 5000 --semilla 2026 --salida datos/
    → datos/train.jsonl (70 %), dev.jsonl (15 %), test.jsonl (15 %)
"""
from __future__ import annotations


def generar(n: int, semilla: int, ruta_gramatica: str, ruta_objetos: str) -> list[dict]:
    """Devuelve n registros que cumplen C5 (validar con ordenred.corpus.validar_registro)."""
    raise NotImplementedError("Rol 5 · generador de corpus (jue 26 – sáb 28)")


def dividir(registros: list[dict], semilla: int = 2026) -> tuple[list[dict], list[dict], list[dict]]:
    """(train 70 %, dev 15 %, test 15 %)."""
    raise NotImplementedError("Rol 5 · división del corpus")


if __name__ == "__main__":
    raise SystemExit("Pendiente (rol 5)")
