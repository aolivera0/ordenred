"""M4 · Etiquetador HMM + Viterbi (contrato C6). Dueño: rol 4.

AIMA §24.1.6. El conjunto de etiquetas son las categorías preterminales de la gramática (C2).
Uso en el pipeline: para cada token <UNK>, el HMM propone su etiqueta y el CYK la usa como preterminal.
"""
from __future__ import annotations


class HMM:
    def entrenar(self, oraciones: list[list[tuple[str, str]]]) -> None:
        """Estima P(etiqueta_t | etiqueta_{t-1}) y P(palabra | etiqueta) por conteo con suavizado.

        `oraciones` es una lista de [(palabra, etiqueta), ...].
        """
        raise NotImplementedError("Rol 4 · HMM.entrenar")

    def viterbi(self, palabras: list[str]) -> tuple[list[str], float]:
        """Secuencia de etiquetas más probable y su log-probabilidad."""
        raise NotImplementedError("Rol 4 · HMM.viterbi")
