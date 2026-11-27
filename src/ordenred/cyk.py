"""
M5b · Analizador CYK probabilístico con k mejores árboles (contrato C6). Dueño: rol 1 (Sergio Nicolás Siabatto Cleves).
AIMA §24.3, fig. 24.5. Log-probabilidades, clausura unaria y punteros de retroceso.

La tabla es tabla[i][j][A] = lista (de mayor a menor logprob, a lo sumo k) de `_Entrada`.
Cada entrada guarda REFERENCIAS a las entradas hijas (no posiciones en listas), así que reordenar
o podar una celda nunca cambia el árbol que representa una entrada ya creada.
"""

from __future__ import annotations

import bisect

from .arbol import Arbol
from .pcfg import PCFG, ReglaPCFG
from .tokenizador import Token


class _Entrada:
    """Una derivación de un no terminal sobre un tramo: su logprob y cómo se construyó."""

    __slots__ = ("logprob", "regla", "hijos", "pos")

    def __init__(
        self,
        logprob: float,
        regla: ReglaPCFG | None,
        hijos: tuple[_Entrada, ...] = (),
        pos: int | None = None,
    ) -> None:
        self.logprob = logprob
        self.regla = regla  # None solo para la etiqueta propuesta por el HMM
        self.hijos = hijos  # () en preterminales; 1 hijo en unarias; 2 en binarias
        self.pos = pos  # posición del terminal (solo preterminales)


def _clave_orden(e: _Entrada) -> float:
    return -e.logprob


class ParserCYK:
    def __init__(self, pcfg: PCFG) -> None:
        self.pcfg = pcfg

    def analizar(
        self,
        terminales: list[str],
        k: int = 1,
        etiquetas_unk: dict[int, str] | None = None,
        tokens: list[Token] | None = None,
    ) -> list[Arbol]:
        """
        Devuelve hasta k árboles con raíz S, distintos y ordenados de mayor a menor probabilidad.

        terminales: `Token.terminal` de cada token ("bloquea", "<CIDR>", ...).
        etiquetas_unk: {posición: preterminal} propuesto por el HMM para palabras fuera del léxico
            (se le asigna log P = 0 a esa etiqueta y se ignora el léxico en esa posición).
        tokens: opcional; si se pasa, las hojas de los árboles llevan esos tokens (con su IP, puerto,
            etc.) en lugar de un token sintético con la forma del terminal. Debe cumplirse
            tokens[i].terminal == terminales[i].
        Lista vacía si la orden no tiene análisis.
        """
        if k < 1:
            raise ValueError("k debe ser ≥ 1")
        n = len(terminales)
        if tokens is not None:
            if len(tokens) != n:
                raise ValueError(f"se dieron {len(tokens)} tokens para {n} terminales")
            for i, (t, term) in enumerate(zip(tokens, terminales)):
                if t.terminal != term:
                    raise ValueError(f"posición {i}: tokens dice {t.terminal!r} y terminales {term!r}")
        if n == 0:
            return []

        tabla: list[list[dict[str, list[_Entrada]]]] = [[{} for _ in range(n)] for _ in range(n)]

        # --- 1. Caso base: tramos de longitud 1 (preterminales) ---
        for i, term in enumerate(terminales):
            celda = tabla[i][i]
            if etiquetas_unk and i in etiquetas_unk:
                celda[etiquetas_unk[i]] = [_Entrada(0.0, None, pos=i)]
            else:
                for regla in self.pcfg.indice_lexico.get(term.lower(), []):
                    self._insertar(celda, regla.izq, _Entrada(regla.logprob, regla, pos=i), k)
            self._clausura_unaria(celda, k)

        # --- 2. Caso recursivo: tramos de longitud 2 a n ---
        for longitud in range(2, n + 1):
            for i in range(n - longitud + 1):
                j = i + longitud - 1
                candidatos: dict[str, list[_Entrada]] = {}
                for corte in range(i, j):
                    celda_izq, celda_der = tabla[i][corte], tabla[corte + 1][j]
                    for B, alternativas_B in celda_izq.items():
                        for C, alternativas_C in celda_der.items():
                            for regla in self.pcfg.indice_binario.get((B, C), []):
                                destino = candidatos.setdefault(regla.izq, [])
                                for eb in alternativas_B:
                                    for ec in alternativas_C:
                                        lp = regla.logprob + eb.logprob + ec.logprob
                                        destino.append(_Entrada(lp, regla, (eb, ec)))
                celda = tabla[i][j]
                for A, lista in candidatos.items():
                    lista.sort(key=_clave_orden)  # estable: los empates conservan el orden de llegada
                    celda[A] = lista[:k]
                self._clausura_unaria(celda, k)

        # --- 3. Reconstrucción ---
        raiz = tabla[0][n - 1].get(self.pcfg.simbolo_inicial, [])
        return [self._reconstruir(self.pcfg.simbolo_inicial, e, terminales, tokens) for e in raiz[:k]]

    @staticmethod
    def _insertar(celda: dict[str, list[_Entrada]], A: str, e: _Entrada, k: int) -> bool:
        """Inserta e entre las k mejores de A. Devuelve False si no entra (y no cambia nada)."""
        lista = celda.setdefault(A, [])
        if len(lista) >= k and e.logprob <= lista[-1].logprob:
            return False
        bisect.insort_right(lista, e, key=_clave_orden)
        if len(lista) > k:
            lista.pop()
        return True

    def _clausura_unaria(self, celda: dict[str, list[_Entrada]], k: int) -> None:
        """Propaga A -> B dentro de la celda manteniendo las k mejores derivaciones de cada A.

        Cada par (regla unaria, entrada hija) se considera una sola vez, así que el ciclo termina
        aunque haya cadenas o ciclos unarios. Una derivación peor que las k que ya tiene A se descarta
        sin marcar cambios (eso era lo que dejaba el ciclo anterior en un bucle infinito).
        """
        pendientes = [(B, e) for B, lista in celda.items() for e in lista]
        vistos: set[tuple[ReglaPCFG, _Entrada]] = set()
        while pendientes:
            B, e = pendientes.pop()
            if e not in celda.get(B, ()):  # ya fue desplazada por mejores derivaciones de B
                continue
            for regla in self.pcfg.indice_unario.get(B, []):
                if (regla, e) in vistos:
                    continue
                vistos.add((regla, e))
                nueva = _Entrada(regla.logprob + e.logprob, regla, (e,))
                if self._insertar(celda, regla.izq, nueva, k):
                    pendientes.append((regla.izq, nueva))

    def _reconstruir(
        self, etiqueta: str, e: _Entrada, terminales: list[str], tokens: list[Token] | None
    ) -> Arbol:
        if not e.hijos:
            if tokens is not None:
                token = tokens[e.pos]
            else:
                term = terminales[e.pos]
                token = Token(texto=term, forma=term, tipo="PALABRA")
            return Arbol(etiqueta=etiqueta, hijos=[], token=token, logprob=e.logprob)
        hijos = [self._reconstruir(r, h, terminales, tokens) for r, h in zip(e.regla.der, e.hijos)]
        return Arbol(etiqueta=etiqueta, hijos=hijos, logprob=e.logprob)
