"""M5a / M6 · Gramática libre de contexto probabilística (contrato C2 y C6). Dueño: rol 1.

AIMA §24.2 (PCFG) y §24.3.2 (aprendizaje de las probabilidades por conteo sobre un treebank).
Formato del archivo: ver gramatica/reglas.pcfg y tests/fixtures/mini.pcfg.
Solo se admiten reglas binarias, léxicas y unarias.
"""
from __future__ import annotations

from .arbol import Arbol


class PCFG:
    @classmethod
    def desde_archivo(cls, ruta: str) -> PCFG:
        """Lee y valida: formato permitido y que las reglas de cada no terminal sumen 1."""
        raise NotImplementedError("Rol 1 · PCFG.desde_archivo")

    def estimar_desde_treebank(self, arboles: list[Arbol], suavizado: float = 0.0) -> None:
        """Máxima verosimilitud: P(A → β) = cuenta(A → β) / cuenta(A), con suavizado aditivo opcional.

        `arbol.reglas(a)` entrega las reglas de cada árbol en la notación del archivo.
        """
        raise NotImplementedError("Rol 1 · PCFG.estimar_desde_treebank")

    def guardar(self, ruta: str) -> None:
        raise NotImplementedError("Rol 1 · PCFG.guardar")
