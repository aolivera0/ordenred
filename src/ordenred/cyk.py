"""M5b · Analizador CYK probabilístico con k mejores árboles (contrato C6). Dueño: rol 1.

AIMA §24.3, fig. 24.5. Tabla NumPy en log-probabilidades, clausura unaria y punteros de retroceso.
"""
from __future__ import annotations

from .arbol import Arbol
from .pcfg import PCFG


class ParserCYK:
    def __init__(self, pcfg: PCFG) -> None:
        self.pcfg = pcfg

    def analizar(
        self,
        terminales: list[str],
        k: int = 1,
        etiquetas_unk: dict[int, str] | None = None,
    ) -> list[Arbol]:
        """Devuelve hasta k árboles con raíz S, ordenados de mayor a menor probabilidad.

        terminales: `Token.terminal` de cada token ("bloquea", "<CIDR>", ...).
        etiquetas_unk: {posición: preterminal} propuesto por el HMM para palabras fuera del léxico.
        Lista vacía si la orden no tiene análisis.
        """
        raise NotImplementedError("Rol 1 · ParserCYK.analizar")
