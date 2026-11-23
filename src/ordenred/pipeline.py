"""Pipeline completo (contrato C7). Dueño: rol 6.

La CLI, la GUI y los experimentos SOLO usan `Pipeline` y `Resultado`.

Orden de las etapas en `traducir` (cada una llena campos de Resultado):
  1. tokenizar (M1)                      → tokens
  2. Corrector.corregir (M2)             → tokens, correcciones        (si config["usar_corrector"])
  3. Clasificador.predecir (M3)          → intencion, prob_intencion
  4. HMM.viterbi sobre los <UNK> (M4)    → etiquetas_unk               (si config["usar_hmm"])
  5. ParserCYK.analizar (M5, PCFG de M6) → arboles (k mejores)
  6. Interprete.interpretar (M7)         → formas_logicas (del mejor árbol)
  7. a_iptables / a_acl_cisco (M8)       → iptables, acl
  + advertencias (orden ambigua, sin análisis, intención y árbol no coinciden…)
  + tiempos_ms por etapa (para E5)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .arbol import Arbol
from .semantica import FormaLogica
from .tokenizador import Token

CONFIG_POR_OMISION: dict = {
    "ruta_gramatica": "src/ordenred/gramatica/reglas.pcfg",
    "ruta_objetos": "src/ordenred/gramatica/objetos.json",
    "dir_modelos": "modelos",
    "usar_corrector": True,
    "suavizado": "interpolacion",  # "laplace" | "backoff" | "interpolacion"
    "clasificador": "nb",  # "nb" | "rl"
    "usar_hmm": True,
    "k": 3,
    "numero_acl": 101,
}


@dataclass
class Resultado:
    texto: str
    tokens: list[Token] = field(default_factory=list)
    correcciones: list[tuple[str, str, float]] = field(default_factory=list)
    intencion: str | None = None
    prob_intencion: dict[str, float] = field(default_factory=dict)
    etiquetas_unk: dict[int, str] = field(default_factory=dict)  # posición → etiqueta del HMM
    arboles: list[Arbol] = field(default_factory=list)  # k mejores
    formas_logicas: list[FormaLogica] = field(default_factory=list)  # del mejor árbol
    iptables: list[str] = field(default_factory=list)
    acl: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    tiempos_ms: dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, config: dict | None = None) -> None:
        desconocidas = set(config or {}) - set(CONFIG_POR_OMISION)
        if desconocidas:
            raise ValueError(f"opciones de configuración desconocidas: {sorted(desconocidas)}")
        self.config = {**CONFIG_POR_OMISION, **(config or {})}

    def traducir(self, texto: str) -> Resultado:
        raise NotImplementedError("Rol 6 · Pipeline.traducir (con stubs: H0; núcleo real: H2)")
