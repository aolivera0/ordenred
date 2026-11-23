"""Tabla de servicios y redes con nombre. Dueño: rol 2.

Archivo: gramatica/objetos.json (formato de ejemplo en tests/fixtures/mini_objetos.json):
    {"servicios": {"ssh": ["tcp", 22], "dns": ["udp", 53], "ping": ["icmp", null]},
     "redes":     {"invitados": "192.168.50.0/24", "servidor web": "10.0.0.10"}}

Interfaz propuesta por el esqueleto; el rol 2 puede ampliarla avisando al grupo.
"""
from __future__ import annotations


class TablaObjetos:
    @classmethod
    def desde_archivo(cls, ruta: str) -> TablaObjetos:
        raise NotImplementedError("Rol 2 · TablaObjetos.desde_archivo")

    def servicio(self, nombre: str) -> tuple[str, int | None] | None:
        """("tcp", 22) para "ssh"; ("icmp", None) para "ping"; None si no existe."""
        raise NotImplementedError("Rol 2 · TablaObjetos.servicio")

    def red(self, nombre: str) -> str | None:
        """IP o CIDR de una red con nombre ("invitados" → "192.168.50.0/24"); None si no existe."""
        raise NotImplementedError("Rol 2 · TablaObjetos.red")
