"""
Árbol sintáctico (contrato C3). Dueño: rol 1 (Sergio Nicolás Siabatto Cleves).

Formato entre corchetes, el mismo del corpus (C5):
    (S (SV (VACC bloquea) (SN (DET el) (NUC (SERV ssh)))))

Un nodo preterminal tiene `token` y no tiene hijos; un nodo interno tiene hijos y `token is None`.
En el texto entre corchetes, la hoja de un preterminal se escribe con la `forma` del token.
Referencia: Russell & Norvig, AIMA 4ta Ed., Cap. 24 (Lenguaje Natural).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .tokenizador import Token

_PIEZAS = re.compile(r"\(|\)|[^\s()]+")


@dataclass
class Arbol:
    etiqueta: str
    hijos: list[Arbol] = field(default_factory=list)
    token: Token | None = None  # solo en preterminales
    logprob: float = 0.0  # log P del subárbol

    @property
    def es_preterminal(self) -> bool:
        return self.token is not None

    def __str__(self) -> str:
        return a_corchetes(self)

    def __repr__(self) -> str:
        return f"Arbol({a_corchetes(self)}, logprob={self.logprob:.4f})"

    def __eq__(self, otro: object) -> bool:
        if not isinstance(otro, Arbol):
            return False
        return (
            self.etiqueta == otro.etiqueta
            and self.hijos == otro.hijos
            and (self.token == otro.token if self.es_preterminal else otro.token is None)
            and abs(self.logprob - otro.logprob) < 1e-6
        )

    def __lt__(self, otro: object) -> bool:
        """
        Soporte para colas de prioridad (heapq) y desempates en algoritmos de k-mejores árboles.
        Compara primero por logprob (mayor probabilidad primero) y luego por serialización canónica.
        """
        if not isinstance(otro, Arbol):
            return NotImplemented
        if abs(self.logprob - otro.logprob) > 1e-9:
            return self.logprob < otro.logprob
        return a_corchetes(self) < a_corchetes(otro)


def a_corchetes(a: Arbol) -> str:
    """Serializa el árbol en el formato canónico de corchetes."""
    if a.es_preterminal:
        if a.hijos:
            raise ValueError(f"el preterminal {a.etiqueta} no puede tener hijos")
        if a.token is None:
            raise ValueError(f"el preterminal {a.etiqueta} debe tener token")
        return f"({a.etiqueta} {a.token.forma})"

    if not a.hijos:
        raise ValueError(f"el nodo interno {a.etiqueta} no tiene hijos ni token")

    return f"({a.etiqueta} " + " ".join(a_corchetes(h) for h in a.hijos) + ")"


def desde_corchetes(s: str, tokens: list[Token] | None = None) -> Arbol:
    """Lee un árbol en formato de corchetes.

    Si se pasan `tokens` (p. ej. la salida de `tokenizar`), se asignan en orden a las hojas y se
    verifica que cada `forma` coincida. Sin `tokens`, cada hoja queda como Token de tipo PALABRA:
    sirve para leer etiquetas y estructura, pero `terminal` no distinguirá "<IP>", "<CIDR>", etc.
    """
    piezas = _PIEZAS.findall(s)
    if not piezas:
        raise ValueError("cadena vacía")

    hojas_recolectadas: list[str] = []

    def nodo(i: int) -> tuple[Arbol, int]:
        if piezas[i] != "(":
            raise ValueError(f"se esperaba '(' en la posición {i}: {piezas[i]!r}")
        if i + 1 >= len(piezas) or piezas[i + 1] in "()":
            raise ValueError(f"falta la etiqueta en la posición {i + 1}")

        etiqueta = piezas[i + 1]
        i += 2

        if i + 1 < len(piezas) and piezas[i] not in "()" and piezas[i + 1] == ")":
            hojas_recolectadas.append(piezas[i])
            pos = len(hojas_recolectadas) - 1
            return Arbol(etiqueta, [], token=_token_hoja(piezas[i], pos)), i + 2

        hijos = []
        while i < len(piezas) and piezas[i] == "(":
            h, i = nodo(i)
            hijos.append(h)

        if not hijos:
            raise ValueError(f"el nodo {etiqueta} no tiene hijos válidos")

        if i >= len(piezas) or piezas[i] != ")":
            raise ValueError(f"se esperaba ')' al cerrar {etiqueta}")

        return Arbol(etiqueta, hijos), i + 1

    def _token_hoja(forma: str, pos: int) -> Token:
        if tokens is None:
            return Token(texto=forma, forma=forma, tipo="PALABRA")
        if pos >= len(tokens):
            raise ValueError(f"el árbol tiene más hojas que los {len(tokens)} tokens dados")
        if tokens[pos].forma != forma:
            raise ValueError(f"hoja {pos}: el árbol dice {forma!r} y el token {tokens[pos].forma!r}")
        return tokens[pos]

    arbol, fin = nodo(0)
    if fin != len(piezas):
        raise ValueError(f"sobra texto después del árbol: {' '.join(piezas[fin:])!r}")

    if tokens is not None and len(hojas_recolectadas) != len(tokens):
        raise ValueError(
            f"el árbol tiene {len(hojas_recolectadas)} hojas y se dieron {len(tokens)} tokens"
        )

    return arbol


def preterminales(a: Arbol) -> list[str]:
    """Etiquetas de los preterminales de izquierda a derecha (las «etiquetas» del HMM)."""
    if a.es_preterminal:
        return [a.etiqueta]
    return [e for h in a.hijos for e in preterminales(h)]


def hojas(a: Arbol) -> list[Token]:
    """Tokens de las hojas de izquierda a derecha."""
    if a.es_preterminal:
        if a.token is None:
            raise ValueError(f"el preterminal {a.etiqueta} no tiene token asignado")
        return [a.token]
    return [t for h in a.hijos for t in hojas(h)]


def reglas(a: Arbol) -> list[tuple[str, tuple[str, ...]]]:
    """Reglas usadas por el árbol, en preorden, con la misma notación de reglas.pcfg (C2).

    Interna:  ("SV", ("VACC", "SN"))     Unaria: ("NUC", ("SERV",))
    Léxica:   ("VACC", ('"bloquea"',))   Especial: ("DIR", ('"<CIDR>"',))
    Es lo que cuenta M6 para estimar la PCFG.
    """
    if a.es_preterminal:
        if a.token is None:
            raise ValueError(f"el preterminal {a.etiqueta} no tiene token asignado")
        return [(a.etiqueta, (f'"{a.token.terminal}"',))]

    propias = [(a.etiqueta, tuple(h.etiqueta for h in a.hijos))]
    return propias + [r for h in a.hijos for r in reglas(h)]