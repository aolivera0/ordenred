"""M7 · Forma lógica e intérprete semántico (contrato C4 y C6). Dueño: rol 2.

AIMA §24.4 (gramáticas aumentadas), §24.4.1 (interpretación semántica composicional)
y §24.4.2 (gramáticas semánticas).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from .arbol import Arbol
from .objetos import TablaObjetos

ACCIONES = ("PERMITIR", "BLOQUEAR", "RECHAZAR", "LIMITAR", "REDIRIGIR", "ELIMINAR", "LISTAR")
PROTOCOLOS = ("tcp", "udp", "icmp", "any")
CADENAS = ("INPUT", "OUTPUT", "FORWARD")


@dataclass
class FormaLogica:
    accion: str
    protocolo: str = "any"
    origen: str | None = None  # IP o CIDR ya resuelto
    destino: str | None = None
    puerto_destino: tuple[int, int] | None = None  # (22, 22) o (1000, 2000)
    puerto_origen: tuple[int, int] | None = None
    cadena: str = "INPUT"
    interfaz: str | None = None
    tasa: str | None = None  # "10/second" (LIMITAR)
    redirigir_a: str | None = None  # "192.168.1.10:8080" (REDIRIGIR)

    def __post_init__(self) -> None:
        if self.accion not in ACCIONES:
            raise ValueError(f"acción inválida: {self.accion!r}; válidas: {ACCIONES}")
        if self.protocolo not in PROTOCOLOS:
            raise ValueError(f"protocolo inválido: {self.protocolo!r}; válidos: {PROTOCOLOS}")
        if self.cadena not in CADENAS:
            raise ValueError(f"cadena inválida: {self.cadena!r}; válidas: {CADENAS}")
        for nombre in ("puerto_destino", "puerto_origen"):
            p = getattr(self, nombre)
            if p is not None:
                p = tuple(p)
                if len(p) != 2 or not (0 <= p[0] <= p[1] <= 65535):
                    raise ValueError(f"{nombre} inválido: {p!r}")
                setattr(self, nombre, p)

    def a_logica(self) -> str:
        """Forma lógica legible: "Bloquear(x) ∧ Origen(x, 10.0.0.0/8) ∧ Puerto(x, 22) ∧ Proto(x, tcp)"."""
        raise NotImplementedError("Rol 2 · FormaLogica.a_logica")

    def a_dict(self) -> dict:
        """Diccionario apto para JSON (C5). Omite los campos que valen None."""
        d = asdict(self)
        for nombre in ("puerto_destino", "puerto_origen"):
            if d[nombre] is not None:
                d[nombre] = list(d[nombre])
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def desde_dict(cls, d: dict) -> FormaLogica:
        validos = {f.name for f in fields(cls)}
        sobrantes = set(d) - validos
        if sobrantes:
            raise ValueError(f"campos desconocidos en la forma lógica: {sorted(sobrantes)}")
        return cls(**d)


class Interprete:
    def __init__(self, objetos: TablaObjetos) -> None:
        self.objetos = objetos

    def interpretar(self, arbol: Arbol) -> list[FormaLogica]:
        """Composición recursiva sobre el árbol. Una orden coordinada («http y ssh») da varias formas lógicas."""
        raise NotImplementedError("Rol 2 · Interprete.interpretar")
