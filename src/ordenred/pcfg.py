"""
M5a / M6 · Gramática libre de contexto probabilística (contrato C2 y C6). Dueño: rol 1 (Sergio Nicolás Siabatto Cleves).

AIMA §24.2 (PCFG) y §24.3.2 (aprendizaje de las probabilidades por conteo sobre un treebank).
Formato del archivo: ver gramatica/reglas.pcfg y tests/fixtures/mini.pcfg.
Solo se admiten reglas binarias, léxicas y unarias.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .arbol import Arbol, reglas


@dataclass(frozen=True)
class ReglaPCFG:
    """Representa una regla A -> der con su probabilidad en escala lineal y logarítmica."""
    izq: str
    der: tuple[str, ...]
    prob: float
    logprob: float

    @property
    def es_binaria(self) -> bool:
        return len(self.der) == 2 and not self.es_lexica

    @property
    def es_unaria(self) -> bool:
        return len(self.der) == 1 and not self.es_lexica

    @property
    def es_lexica(self) -> bool:
        return len(self.der) == 1 and self.der[0].startswith('"') and self.der[0].endswith('"')

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

    def agregar_regla(self, izq: str, der: tuple[str, ...], prob: float) -> None:
        """Registra e indexa una regla asegurando logprob válido."""
        if prob <= 0.0:
            raise ValueError(f"Probabilidad inválida ({prob}) en regla: {izq} -> {der}")

        logprob = math.log(prob)
        regla = ReglaPCFG(izq=izq, der=der, prob=prob, logprob=logprob)

        self.reglas.append(regla)
        self._por_no_terminal[izq].append(regla)

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

    def validar_consistencia(self, tolerancia: float = 1e-3) -> None:
        """Verifica que para cada no terminal A, sum(P(A -> der)) == 1.0 (AIMA §24.2)."""
        for no_terminal, lista_reglas in self._por_no_terminal.items():
            suma = sum(r.prob for r in lista_reglas)
            if abs(suma - 1.0) > tolerancia:
                raise ValueError(
                    f"Las reglas de '{no_terminal}' no suman 1.0 (suma actual: {suma:.6f})"
                )

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

                if "->" not in linea:
                    raise ValueError(f"Línea {num_linea} no contiene '->': {linea}")

                partes = linea.split("->")
                izq = partes[0].strip()
                resto = partes[1].strip()

                if not ("[" in resto and resto.endswith("]")):
                    raise ValueError(f"Línea {num_linea} no tiene corchetes de probabilidad: {linea}")

                der_str, prob_str = resto.rsplit("[", 1)
                prob_str = prob_str.rstrip("]").strip()
                try:
                    prob = float(prob_str)
                except ValueError:
                    raise ValueError(f"Línea {num_linea}: probabilidad inválida '{prob_str}'")

                der = tuple(der_str.strip().split())
                instancia.agregar_regla(izq, der, prob)

        instancia.validar_consistencia()
        return instancia

    def estimar_desde_treebank(self, arboles: list[Arbol], suavizado: float = 0.0) -> None:
        """Máxima verosimilitud: P(A → β) = cuenta(A → β) / cuenta(A), con suavizado aditivo opcional.

        `arbol.reglas(a)` entrega las reglas de cada árbol en la notación del archivo.
        """
        if not arboles:
            return

        conteo_reglas: dict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
        conteo_izq: dict[str, int] = defaultdict(int)
        vocab_por_izq: dict[str, set[tuple[str, ...]]] = defaultdict(set)

        for a in arboles:
            for izq, der in reglas(a):
                conteo_reglas[(izq, der)] += 1
                conteo_izq[izq] += 1
                vocab_por_izq[izq].add(der)

        # Reiniciar gramática e índices
        self.reglas.clear()
        self.indice_binario.clear()
        self.indice_unario.clear()
        self.indice_lexico.clear()
        self._por_no_terminal.clear()

        # Calcular probabilidades MLE
        for (izq, der), c_regla in conteo_reglas.items():
            c_izq = conteo_izq[izq]
            v_size = len(vocab_por_izq[izq])

            if suavizado > 0.0:
                prob = (c_regla + suavizado) / (c_izq + suavizado * v_size)
            else:
                prob = c_regla / c_izq

            self.agregar_regla(izq, der, prob)

        self.validar_consistencia()

    def guardar(self, ruta: str | Path) -> None:
        """Escribe las reglas en el formato estándar C2."""
        ruta_p = Path(ruta)
        ruta_p.parent.mkdir(parents=True, exist_ok=True)

        with open(ruta_p, "w", encoding="utf-8") as f:
            f.write("# Reglas PCFG generadas automáticamente\n")
            for r in self.reglas:
                der_str = " ".join(r.der)
                f.write(f"{r.izq} -> {der_str} [{r.prob:.6f}]\n")