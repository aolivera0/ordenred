"""M8 · Forma lógica → reglas de firewall (contrato C6). Dueño: rol 2.

Salida principal: iptables. Secundaria: ACL extendida numerada de Cisco IOS
(CIDR → máscara wildcard, eq / range, permit / deny).
No es una técnica del capítulo 24: es el «pegamento» de salida.
"""
from __future__ import annotations

from .semantica import FormaLogica


def a_iptables(fls: list[FormaLogica]) -> list[str]:
    """Ej.: [BLOQUEAR tcp origen 10.0.0.0/8 puerto 22] → ["iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 22 -j DROP"]."""
    raise NotImplementedError("Rol 2 · a_iptables")


def a_acl_cisco(fls: list[FormaLogica], numero: int = 101) -> list[str]:
    """Ej.: → ["access-list 101 deny tcp 10.0.0.0 0.255.255.255 any eq 22"]."""
    raise NotImplementedError("Rol 2 · a_acl_cisco")
