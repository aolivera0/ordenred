"""Inyección de errores tipográficos controlados (para el corrector y E2). Dueño: rol 5.

Operaciones: sustitución por tecla vecina (teclado QWERTY español), omisión, inserción y
transposición. No altera IP, CIDR, números ni interfaces.
"""
from __future__ import annotations

import random


def agregar_ruido(texto: str, tasa: float, rng: random.Random) -> str:
    """Altera aproximadamente `tasa` de los caracteres alterables (0.0 ≤ tasa ≤ 1.0)."""
    raise NotImplementedError("Rol 5 · ruido (vie 27 – sáb 28)")
