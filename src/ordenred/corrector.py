"""M2b · Corrector ortográfico por canal ruidoso (contrato C6). Dueño: rol 3.

Elige la palabra w que maximiza P(w | contexto) · P(lo escrito | w):
P(w | contexto) del modelo de n-gramas de palabras; P(lo escrito | w) por tipo de edición
(distancia de Damerau-Levenshtein implementada a mano). AIMA §24.1.2–24.1.4.
No corrige tokens IP, CIDR, NUM, RANGO ni IFAZ.
"""
from __future__ import annotations

from .ngramas import ModeloNGramas
from .tokenizador import Token


class Corrector:
    def __init__(
        self,
        vocabulario: set[str],
        lm_palabras: ModeloNGramas,
        lm_caracteres: ModeloNGramas,
    ) -> None:
        self.vocabulario = vocabulario
        self.lm_palabras = lm_palabras
        self.lm_caracteres = lm_caracteres

    def corregir(self, tokens: list[Token]) -> tuple[list[Token], list[tuple[str, str, float]]]:
        """Devuelve (tokens corregidos, [(antes, después, logprob de la corrección)])."""
        raise NotImplementedError("Rol 3 · Corrector.corregir")
