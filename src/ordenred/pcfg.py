"""
M5a / M6 · Gramática libre de contexto probabilística (contrato C2 y C6). Dueño: rol 1 (Sergio Nicolás Siabatto Cleves).

AIMA §24.2 (PCFG) y §24.3.2 (aprendizaje de las probabilidades por conteo sobre un treebank).
Formato del archivo: ver gramatica/reglas.pcfg y tests/fixtures/mini.pcfg.
Solo se admiten reglas binarias, léxicas y unarias.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .arbol import Arbol, reglas

# A -> der [p]   con comentario opcional al final:  A -> der [p]  # nota
_LINEA = re.compile(r"^(?P<izq>\S+)\s*->\s*(?P<der>.+?)\s*\[\s*(?P<prob>[^\]]+?)\s*\]\s*(?:#.*)?$")


def _es_terminal(simbolo: str) -> bool:
    return len(simbolo) >= 2 and simbolo.startswith('"') and simbolo.endswith('"')


@dataclass(frozen=True)
class ReglaPCFG:
    """Representa una regla A -> der con su probabilidad en escala lineal y logarítmica."""

    izq: str
    der: tuple[str, ...]
    prob: float
    logprob: float

    @property
    def es_binaria(self) -> bool:
        return len(self.der) == 2 and not any(_es_terminal(s) for s in self.der)

    @property
    def es_unaria(self) -> bool:
        return len(self.der) == 1 and not _es_terminal(self.der[0])

    @property
    def es_lexica(self) -> bool:
        return len(self.der) == 1 and _es_terminal(self.der[0])

    @property
    def terminal_lexico(self) -> str:
        """Devuelve el texto terminal sin comillas (ej. 'bloquea' o '<CIDR>')."""
        if self.es_lexica:
            return self.der[0][1:-1]
        raise ValueError(f"La regla {self.izq} -> {self.der} no es léxica")


class PCFG:
    def __init__(self, simbolo_inicial: str = "S") -> None:
        self.simbolo_inicial = simbolo_inicial
        self.reglas: list[ReglaPCFG] = []

        # Índices optimizados para consulta O(1) en el CYK:
        # (B, C) -> [ReglaPCFG(A -> B C), ...]
        self.indice_binario: dict[tuple[str, str], list[ReglaPCFG]] = defaultdict(list)
        # B -> [ReglaPCFG(A -> B), ...]
        self.indice_unario: dict[str, list[ReglaPCFG]] = defaultdict(list)
        # "terminal" (en minúsculas o <TAG>) -> [ReglaPCFG(A -> "terminal"), ...]
        self.indice_lexico: dict[str, list[ReglaPCFG]] = defaultdict(list)
        # Diccionario para verificación de consistencia sum(P) == 1.0
        self._por_no_terminal: dict[str, list[ReglaPCFG]] = defaultdict(list)

    def _reiniciar(self) -> None:
        self.reglas.clear()
        self.indice_binario.clear()
        self.indice_unario.clear()
        self.indice_lexico.clear()
        self._por_no_terminal.clear()

    def agregar_regla(self, izq: str, der: tuple[str, ...], prob: float) -> None:
        """Registra e indexa una regla asegurando formato y logprob válidos."""
        if not (0.0 < prob <= 1.0):
            raise ValueError(f"Probabilidad inválida ({prob}) en regla: {izq} -> {der}")
        if _es_terminal(izq) or not izq:
            raise ValueError(f"El lado izquierdo debe ser un no terminal: {izq!r}")
        if any((r.izq, r.der) == (izq, der) for r in self._por_no_terminal.get(izq, ())):
            raise ValueError(f"Regla duplicada: {izq} -> {' '.join(der)}")

        regla = ReglaPCFG(izq=izq, der=der, prob=prob, logprob=math.log(prob))
        if regla.es_lexica:
            self.indice_lexico[regla.terminal_lexico.lower()].append(regla)
        elif regla.es_unaria:
            self.indice_unario[regla.der[0]].append(regla)
        elif regla.es_binaria:
            self.indice_binario[(regla.der[0], regla.der[1])].append(regla)
        else:
            raise ValueError(
                f"Formato no permitido en regla (debe ser binaria, unaria o léxica): {izq} -> {der}"
            )
        self.reglas.append(regla)
        self._por_no_terminal[izq].append(regla)

    def validar_consistencia(self, tolerancia: float = 1e-3) -> None:
        """Verifica la gramática (AIMA §24.2):

        - para cada no terminal A, sum(P(A -> der)) == 1.0;
        - todo símbolo sin comillas del lado derecho tiene sus propias reglas (si no, casi siempre es
          un terminal al que le faltan las comillas);
        - existe el símbolo inicial.
        """
        for no_terminal, lista_reglas in self._por_no_terminal.items():
            suma = sum(r.prob for r in lista_reglas)
            if abs(suma - 1.0) > tolerancia:
                raise ValueError(
                    f"Las reglas de '{no_terminal}' no suman 1.0 (suma actual: {suma:.6f})"
                )
        indefinidos = sorted(
            {s for r in self.reglas if not r.es_lexica for s in r.der} - set(self._por_no_terminal)
        )
        if indefinidos:
            raise ValueError(f"Símbolos usados sin reglas propias (¿faltan comillas?): {indefinidos}")
        if self.reglas and self.simbolo_inicial not in self._por_no_terminal:
            raise ValueError(f"No hay reglas para el símbolo inicial {self.simbolo_inicial!r}")

    @classmethod
    def desde_archivo(cls, ruta: str | Path) -> PCFG:
        """Lee y valida: formato permitido y que las reglas de cada no terminal sumen 1."""
        instancia = cls()
        ruta_p = Path(ruta)
        if not ruta_p.exists():
            raise FileNotFoundError(f"Archivo de gramática no encontrado: {ruta}")

        with open(ruta_p, "r", encoding="utf-8") as f:
            for num_linea, linea in enumerate(f, 1):
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                m = _LINEA.match(linea)
                if m is None:
                    raise ValueError(
                        f"Línea {num_linea}: se esperaba «A -> der [p]», se encontró: {linea}"
                    )
                try:
                    prob = float(m["prob"])
                except ValueError:
                    raise ValueError(f"Línea {num_linea}: probabilidad inválida '{m['prob']}'") from None
                try:
                    instancia.agregar_regla(m["izq"], tuple(m["der"].split()), prob)
                except ValueError as e:
                    raise ValueError(f"Línea {num_linea}: {e}") from None

        instancia.validar_consistencia()
        return instancia

    def estimar_desde_treebank(self, arboles: list[Arbol], suavizado: float = 0.0) -> None:
        """Máxima verosimilitud con suavizado aditivo opcional (AIMA §24.3.2):

            P(A → β) = (cuenta(A → β) + α) / (cuenta(A) + α · |R_A|)

        R_A son las reglas de A en la gramática actual más las nuevas que aparezcan en el treebank,
        así que con α > 0 las reglas de la gramática que no se vieron conservan algo de probabilidad
        (y el analizador no pierde cobertura). Con α = 0 se descartan, como manda la MLE.
        Los no terminales que no aparecen en el treebank conservan sus probabilidades.

        Los árboles deben leerse con sus tokens (`desde_corchetes(s, tokenizar(orden))`): sin ellos,
        una IP queda como la palabra "10.0.0.5" y no como el terminal "<IP>".
        """
        if suavizado < 0:
            raise ValueError("el suavizado debe ser ≥ 0")
        if not arboles:
            return

        conteo_reglas: dict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
        conteo_izq: dict[str, int] = defaultdict(int)
        for a in arboles:
            for izq, der in reglas(a):
                conteo_reglas[(izq, der)] += 1
                conteo_izq[izq] += 1

        # Soporte de cada no terminal: reglas existentes (en su orden) + nuevas observadas.
        soporte: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        anteriores: dict[tuple[str, tuple[str, ...]], float] = {}
        for r in self.reglas:
            soporte[r.izq].append(r.der)
            anteriores[(r.izq, r.der)] = r.prob
        for izq, der in conteo_reglas:
            if der not in soporte[izq]:
                soporte[izq].append(der)

        self._reiniciar()
        for izq, ders in soporte.items():
            c_izq = conteo_izq.get(izq, 0)
            for der in ders:
                if c_izq == 0:
                    prob = anteriores[(izq, der)]
                else:
                    prob = (conteo_reglas.get((izq, der), 0) + suavizado) / (c_izq + suavizado * len(ders))
                if prob > 0.0:
                    self.agregar_regla(izq, der, prob)

        self.validar_consistencia()

    def guardar(self, ruta: str | Path) -> None:
        """Escribe las reglas en el formato estándar C2, con precisión suficiente para releerlas."""
        ruta_p = Path(ruta)
        ruta_p.parent.mkdir(parents=True, exist_ok=True)

        with open(ruta_p, "w", encoding="utf-8") as f:
            f.write("# Reglas PCFG generadas automáticamente\n")
            for r in self.reglas:
                der_str = " ".join(r.der)
                f.write(f"{r.izq} -> {der_str} [{r.prob:.12g}]\n")
