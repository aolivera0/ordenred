"""Interfaz de línea de comandos. Dueño: rol 6.

Uso previsto:
    PYTHONPATH=src python -m ordenred.cli traducir "bloquea el ssh desde 10.0.0.0/8" --formato ambos --arbol
    PYTHONPATH=src python -m ordenred.cli lote ordenes.txt
    PYTHONPATH=src python -m ordenred.cli entrenar
"""
from __future__ import annotations

import argparse
import sys


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ordenred", description="Traduce órdenes en español a reglas de firewall.")
    sub = p.add_subparsers(dest="comando", required=True)

    t = sub.add_parser("traducir", help="traduce una orden")
    t.add_argument("orden")
    t.add_argument("--formato", choices=["iptables", "acl", "ambos"], default="ambos")
    t.add_argument("--k", type=int, default=3, help="número de árboles alternativos")
    t.add_argument("--sin-corrector", action="store_true")
    t.add_argument("--clasificador", choices=["nb", "rl"], default="nb")
    t.add_argument("--arbol", action="store_true", help="muestra el árbol sintáctico")

    l = sub.add_parser("lote", help="traduce un archivo con una orden por línea")
    l.add_argument("archivo")

    sub.add_parser("entrenar", help="reentrena los modelos y los guarda en modelos/")
    return p


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    print(f"Comando «{args.comando}» pendiente de implementar (rol 6).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
