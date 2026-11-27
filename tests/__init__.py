"""Pruebas. Ejecutar desde la raíz del proyecto:  python -m unittest -v"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FIXTURES = RAIZ / "tests" / "fixtures"
sys.path.insert(0, str(RAIZ / "src"))


def tokens_fixture(formas: list[str]):
    """Tokens con tipo para las órdenes de los fixtures, mientras no exista `tokenizar` (rol 5).

    Solo distingue IP, CIDR y NUM; suficiente para tests/fixtures/mini_corpus.jsonl.
    """
    import re

    from ordenred.tokenizador import Token

    def tipo(f: str) -> str:
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+/\d+", f):
            return "CIDR"
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", f):
            return "IP"
        if re.fullmatch(r"\d+", f):
            return "NUM"
        return "PALABRA"

    return [Token(f, f, tipo(f)) for f in formas]
