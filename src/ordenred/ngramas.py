"""M2a · Modelos de n-gramas con suavizado (contrato C6). Dueño: rol 3.

AIMA §24.1.2 (n-gramas como cadenas de Markov de orden n−1), §24.1.3 (modelos de caracteres)
y §24.1.4 (suavizado: Laplace, backoff, interpolación lineal; <UNK>).
Todas las probabilidades se devuelven en logaritmo natural.
"""
from __future__ import annotations

SUAVIZADOS = ("laplace", "backoff", "interpolacion")
UNIDADES = ("palabra", "caracter")


class ModeloNGramas:
    def __init__(self, n: int, suavizado: str = "laplace", unidad: str = "palabra") -> None:
        if n < 1:
            raise ValueError("n debe ser ≥ 1")
        if suavizado not in SUAVIZADOS:
            raise ValueError(f"suavizado inválido: {suavizado!r}; válidos: {SUAVIZADOS}")
        if unidad not in UNIDADES:
            raise ValueError(f"unidad inválida: {unidad!r}; válidas: {UNIDADES}")
        self.n = n
        self.suavizado = suavizado
        self.unidad = unidad

    def entrenar(self, secuencias: list[list[str]]) -> None:
        """Cuenta n-gramas (con marcas de inicio y fin) sobre secuencias de palabras o de caracteres."""
        raise NotImplementedError("Rol 3 · ModeloNGramas.entrenar")

    def ajustar_lambdas(self, dev: list[list[str]]) -> None:
        """Interpolación: busca λ_n + … + λ_1 = 1 que minimicen la perplejidad en `dev`."""
        raise NotImplementedError("Rol 3 · ModeloNGramas.ajustar_lambdas")

    def logprob(self, secuencia: list[str]) -> float:
        raise NotImplementedError("Rol 3 · ModeloNGramas.logprob")

    def perplejidad(self, secuencias: list[list[str]]) -> float:
        raise NotImplementedError("Rol 3 · ModeloNGramas.perplejidad")
