"""M3 · Clasificación de la intención (contrato C6). Dueño: rol 4.

Naive Bayes con bolsa de palabras (generativo, AIMA §24.1.1) frente a regresión logística
multiclase (discriminativa, AIMA §24.1.6). Las clases son las acciones de FormaLogica.
"""
from __future__ import annotations

from typing import Protocol


class Clasificador(Protocol):
    def entrenar(self, X: list[list[str]], y: list[str]) -> None: ...

    def predecir(self, tokens: list[str]) -> tuple[str, dict[str, float]]:
        """Devuelve (clase más probable, {clase: P(clase | tokens)}) con probabilidades que suman 1."""
        ...


class NaiveBayes:
    def __init__(self, alfa: float = 1.0) -> None:
        self.alfa = alfa  # suavizado de Laplace (aditivo)

    def entrenar(self, X: list[list[str]], y: list[str]) -> None:
        raise NotImplementedError("Rol 4 · NaiveBayes.entrenar")

    def predecir(self, tokens: list[str]) -> tuple[str, dict[str, float]]:
        raise NotImplementedError("Rol 4 · NaiveBayes.predecir")


class RegresionLogistica:
    def __init__(self, tasa: float = 0.1, epocas: int = 200, l2: float = 1e-3, semilla: int = 2026) -> None:
        self.tasa = tasa
        self.epocas = epocas
        self.l2 = l2
        self.semilla = semilla

    def entrenar(self, X: list[list[str]], y: list[str]) -> None:
        raise NotImplementedError("Rol 4 · RegresionLogistica.entrenar")

    def predecir(self, tokens: list[str]) -> tuple[str, dict[str, float]]:
        raise NotImplementedError("Rol 4 · RegresionLogistica.predecir")
