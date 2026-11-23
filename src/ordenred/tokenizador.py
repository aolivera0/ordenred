"""M1 · Tokenizador (contrato C1). Dueño: rol 5.

AIMA §24.1.1 (tokenización) y §24.1.4 (palabras desconocidas, <UNK>).
"""
from __future__ import annotations

from dataclasses import dataclass

TIPOS = ("PALABRA", "IP", "CIDR", "NUM", "RANGO", "IFAZ")


@dataclass(frozen=True)
class Token:
    """Unidad mínima que circula por todo el pipeline.

    texto: tal como lo escribió el usuario ("Bloqueá", "10.0.0.0/8").
    forma: normalizada, en minúsculas y sin tildes ("bloquea", "10.0.0.0/8").
    tipo:  uno de TIPOS.
    """

    texto: str
    forma: str
    tipo: str

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS:
            raise ValueError(f"tipo de token desconocido: {self.tipo!r}; válidos: {TIPOS}")

    @property
    def terminal(self) -> str:
        """Símbolo que ve la gramática: la forma si es PALABRA; si no, "<IP>", "<CIDR>", "<NUM>", "<RANGO>" o "<IFAZ>"."""
        return self.forma if self.tipo == "PALABRA" else f"<{self.tipo}>"


def tokenizar(texto: str) -> list[Token]:
    """Divide una orden en tokens.

    - Reconoce IP (10.0.0.5), CIDR (10.0.0.0/8), NUM (22), RANGO (1000-2000 o 1000:2000)
      e IFAZ (eth0, wlan1, enp3s0, Gi0/1).
    - Normaliza: minúsculas, sin tildes.
    - Separa contracciones: «al» → «a el», «del» → «de el».
    """
    raise NotImplementedError("Rol 5 · tokenizar (H0: martes 24)")
