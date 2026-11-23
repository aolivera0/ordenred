"""Ejecutor de escenarios E1–E5. Dueño: rol 7.

Corre un escenario con varias semillas (≥ 5), guarda CSV en experimentos/resultados/ y calcula
media e intervalo de confianza del 95 %. Las figuras (matplotlib) van a experimentos/figuras/.

Uso previsto:
    PYTHONPATH=src python experimentos/runner.py E2 --semillas 5
"""
from __future__ import annotations

ESCENARIOS = ("E1", "E2", "E3", "E4", "E5")


def correr(escenario: str, semillas: list[int]) -> list[dict]:
    raise NotImplementedError("Rol 7 · runner (mié 25 – sáb 28)")


if __name__ == "__main__":
    raise SystemExit("Pendiente (rol 7)")
