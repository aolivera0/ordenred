"""M4 · Etiquetador HMM + Viterbi (contrato C6). Dueño: rol 4.

AIMA 4.ª ed. global, §24.1.6, pp. 880–883: el etiquetado gramatical como modelo oculto de Markov
(las etiquetas son los estados ocultos y las palabras las observaciones) y el algoritmo de Viterbi
para hallar la secuencia de etiquetas más probable (p. 881). El HMM es un modelo **generativo**:
modela la probabilidad conjunta P(palabras, etiquetas) = Π P(etiqueta_t | etiqueta_{t-1}) ·
P(palabra_t | etiqueta_t), a diferencia de un modelo discriminativo como la regresión logística,
que modela directamente P(etiqueta | palabras) (pp. 882–883).

El conjunto de etiquetas son las categorías preterminales de la gramática (C2), aprendidas del corpus.
Uso en el pipeline: para cada token fuera del léxico de la PCFG, el HMM etiqueta la oración completa y
`etiquetas_unk` devuelve la etiqueta propuesta para esas posiciones; el CYK la usa como preterminal.

Decisiones de diseño
--------------------
- Entrada: los **terminales** (`Token.terminal`: "bloquea", "<CIDR>", "<NUM>"...). Las palabras se usan
  tal como llegan (el tokenizador ya normaliza a minúsculas y sin tildes).
- Estados especiales INICIO y FIN (este último opcional, `usar_fin`). Con FIN la probabilidad de la
  oración incluye P(FIN | última etiqueta), así que las secuencias de distinta longitud compiten en
  igualdad de condiciones y cada fila de transiciones (más su probabilidad de FIN) suma 1.
- Palabras raras (**se reemplazan**, que es el estándar): en el entrenamiento, una palabra cuya
  frecuencia total es ≤ `umbral_raras` NO se cuenta como ella misma sino solo como su firma
  (`firma(w)`: "<UNK-DIG>", "<UNK-sufijo>" o "<UNK>"). Así las firmas aprenden qué etiquetas suelen
  tener las palabras poco frecuentes, que es lo más parecido a una palabra nunca vista.
  Consecuencia: una palabra rara del entrenamiento no es «conocida» (`es_conocida` da False) y en la
  predicción se trata igual que una palabra nueva (por su firma).
- Los terminales especiales ("<IP>", "<CIDR>", "<NUM>", "<RANGO>", "<IFAZ>") son palabras normales y
  **nunca se reemplazan por su firma** aunque sean raros: ya son una clase de palabras y su etiqueta
  (DIR, NUMERO...) es muy estable. Si uno de ellos no se vio nunca, en la predicción cae en "<UNK>".
- Suavizado aditivo (Laplace, α > 0; α = 1 por defecto) en transiciones y emisiones: ninguna
  probabilidad es 0, así que el único -inf de Viterbi es el de la máscara de `permitidas`.
  Probabilidades en logaritmo.
- Persistencia (acuerdo C6): JSON con conteos enteros; al cargar se recalculan las tablas de log, así
  que la ida y vuelta da resultados idénticos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._persistencia import VERSION, escribir_json, leer_json

TERMINALES_ESPECIALES = frozenset({"<IP>", "<CIDR>", "<NUM>", "<RANGO>", "<IFAZ>"})
FIRMA_DESCONOCIDA = "<UNK>"
FIRMA_DIGITO = "<UNK-DIG>"
# Sufijos del español que delatan la categoría (verbos, sustantivos, adverbios, plurales...).
SUFIJOS = ("ciones", "cion", "mente", "ando", "iendo", "ar", "er", "ir", "as", "os", "es", "s", "a", "e", "o")
_SUFIJOS_POR_LONGITUD = tuple(sorted(SUFIJOS, key=len, reverse=True))  # sort estable: el más largo primero


def firma(palabra: str) -> str:
    """Clase de forma de una palabra desconocida, para las emisiones del HMM (AIMA §24.1.6, pp. 880–881).

    "<UNK-DIG>" si contiene un dígito; si no, "<UNK-" + sufijo + ">" con el sufijo más largo de
    `SUFIJOS` en que termina la palabra; si no termina en ninguno, "<UNK>".
    """
    if any(c.isdigit() for c in palabra):
        return FIRMA_DIGITO
    for sufijo in _SUFIJOS_POR_LONGITUD:
        if palabra.endswith(sufijo):
            return f"<UNK-{sufijo}>"
    return FIRMA_DESCONOCIDA


class HMM:
    """Modelo oculto de Markov de primer orden para etiquetado gramatical (AIMA §24.1.6, pp. 880–883).

    alfa_trans: suavizado aditivo (α > 0) de las probabilidades iniciales, de transición y de FIN.
    alfa_emis: suavizado aditivo (Laplace, α > 0) de las emisiones P(palabra | etiqueta).
    umbral_raras: las palabras con frecuencia total ≤ umbral se entrenan como su firma (0 = ninguna).
    usar_fin: si True, se modela la transición a un estado FIN tras la última palabra.
    """

    TIPO = "HMM"
    # Campos que debe traer el JSON de `guardar`, además de "tipo" y "version".
    CAMPOS = (
        "alfa_trans", "alfa_emis", "umbral_raras", "usar_fin", "etiquetas", "n_oraciones", "conteo_inicio",
        "conteo_trans", "conteo_fin", "conteo_etiqueta", "conteo_emision", "conocidas", "firmas",
    )

    def __init__(
        self,
        alfa_trans: float = 1.0,
        alfa_emis: float = 1.0,
        umbral_raras: int = 1,
        usar_fin: bool = True,
    ) -> None:
        if not (0 < alfa_trans < float("inf") and 0 < alfa_emis < float("inf")):
            raise ValueError("los parámetros de suavizado deben ser > 0 y finitos (Laplace)")
        if umbral_raras < 0:
            raise ValueError("umbral_raras debe ser ≥ 0")
        self.alfa_trans = float(alfa_trans)
        self.alfa_emis = float(alfa_emis)
        self.umbral_raras = int(umbral_raras)
        self.usar_fin = bool(usar_fin)
        self._entrenado = False
        # Tabla δ y punteros ψ de la última decodificación (para inspección y la sustentación).
        self.delta: np.ndarray | None = None
        self.psi: np.ndarray | None = None

    # ------------------------------------------------------------------ entrenamiento

    def entrenar(self, oraciones: list[list[tuple[str, str]]]) -> None:
        """Estima P(etiqueta_t | etiqueta_{t-1}) y P(palabra | etiqueta) por conteo con suavizado.

        AIMA §24.1.6, pp. 880–881: las probabilidades de transición y de emisión se estiman contando en un
        corpus etiquetado. `oraciones` es una lista de [(palabra, etiqueta), ...]. Las oraciones vacías se
        ignoran; si no queda ninguna, ValueError.

        Fórmulas (α = alfa_trans, T = número de etiquetas, N = número de oraciones):
            π_j   = (C(INICIO, j) + α) / (N + α·T)
            con FIN:  A_ij  = (C(i, j) + α) / (C(i) + α·(T+1)),  fin_i = (C(i, FIN) + α) / (C(i) + α·(T+1))
            sin FIN:  A_ij  = (C(i, j) + α) / (C(i → ·) + α·T),  C(i → ·) = Σ_j C(i, j)
            B_t(w) = (C(t, w) + α_e) / (C(t) + α_e·|V′|),  V′ = conocidas ∪ firmas vistas ∪ {<UNK>}
        """
        oraciones = [list(o) for o in oraciones if len(o) > 0]
        if not oraciones:
            raise ValueError("no hay oraciones para entrenar el HMM")

        frecuencia: dict[str, int] = {}
        etiquetas: set[str] = set()
        for oracion in oraciones:
            for palabra, etiqueta in oracion:
                frecuencia[palabra] = frecuencia.get(palabra, 0) + 1
                etiquetas.add(etiqueta)

        self.etiquetas = sorted(etiquetas)
        self.n_oraciones = len(oraciones)
        self.conteo_inicio = {t: 0 for t in self.etiquetas}
        self.conteo_trans = {i: {j: 0 for j in self.etiquetas} for i in self.etiquetas}
        self.conteo_fin = {t: 0 for t in self.etiquetas}
        self.conteo_etiqueta = {t: 0 for t in self.etiquetas}
        self.conteo_emision: dict[str, dict[str, int]] = {t: {} for t in self.etiquetas}
        self.conocidas: set[str] = set()
        self.firmas: set[str] = set()

        for oracion in oraciones:
            anterior = None
            for palabra, etiqueta in oracion:
                observacion = self._observacion_entrenamiento(palabra, frecuencia[palabra])
                if anterior is None:
                    self.conteo_inicio[etiqueta] += 1
                else:
                    self.conteo_trans[anterior][etiqueta] += 1
                self.conteo_etiqueta[etiqueta] += 1
                emisiones = self.conteo_emision[etiqueta]
                emisiones[observacion] = emisiones.get(observacion, 0) + 1
                anterior = etiqueta
            self.conteo_fin[anterior] += 1

        self._calcular_tablas()

    def _observacion_entrenamiento(self, palabra: str, frecuencia: int) -> str:
        """La palabra, o su firma si es rara (reemplazo de hapax; los terminales especiales no se firman)."""
        if frecuencia <= self.umbral_raras and palabra not in TERMINALES_ESPECIALES:
            f = firma(palabra)
            self.firmas.add(f)
            return f
        self.conocidas.add(palabra)
        return palabra

    def _calcular_tablas(self) -> None:
        """Convierte los conteos enteros en las tablas de log-probabilidades (π, A, fin, B)."""
        T = len(self.etiquetas)
        self.indice_etiqueta = {t: k for k, t in enumerate(self.etiquetas)}
        self.vocabulario = sorted(self.conocidas | self.firmas | {FIRMA_DESCONOCIDA})
        self.indice_obs = {w: k for k, w in enumerate(self.vocabulario)}
        a_t, a_e = self.alfa_trans, self.alfa_emis

        inicio = np.array([self.conteo_inicio[t] for t in self.etiquetas], dtype=float)
        trans = np.array([[self.conteo_trans[i][j] for j in self.etiquetas] for i in self.etiquetas], dtype=float)
        fin = np.array([self.conteo_fin[t] for t in self.etiquetas], dtype=float)
        por_etiqueta = np.array([self.conteo_etiqueta[t] for t in self.etiquetas], dtype=float)

        # Con α > 0 ningún numerador ni denominador es 0, así que todos los logaritmos son finitos.
        self.log_pi = np.log((inicio + a_t) / (self.n_oraciones + a_t * T))
        if self.usar_fin:
            # Cada aparición de i va seguida de una etiqueta o de FIN: C(i) = Σ_j C(i, j) + C(i, FIN).
            den = por_etiqueta + a_t * (T + 1)
            self.log_A = np.log((trans + a_t) / den[:, None])
            self.log_fin = np.log((fin + a_t) / den)
        else:
            salidas = np.array([sum(self.conteo_trans[i].values()) for i in self.etiquetas], dtype=float)
            self.log_A = np.log((trans + a_t) / (salidas + a_t * T)[:, None])
            self.log_fin = np.zeros(T)  # sin FIN no se suma nada al final

        emision = np.zeros((T, len(self.vocabulario)))
        for t, conteos in self.conteo_emision.items():
            for w, c in conteos.items():
                emision[self.indice_etiqueta[t], self.indice_obs[w]] = c
        self.log_B = np.log((emision + a_e) / (por_etiqueta + a_e * len(self.vocabulario))[:, None])
        self._entrenado = True

    # ------------------------------------------------------------------ observaciones

    def es_conocida(self, palabra: str) -> bool:
        """True si la palabra se entrenó como ella misma (no como firma de palabra rara)."""
        self._exigir_entrenado()
        return palabra in self.conocidas

    def observacion(self, palabra: str) -> str:
        """Símbolo de V′ que representa a la palabra: ella misma, su firma o "<UNK>"."""
        self._exigir_entrenado()
        if palabra in self.conocidas:
            return palabra
        f = firma(palabra)
        return f if f in self.indice_obs else FIRMA_DESCONOCIDA

    def _indices_obs(self, palabras: list[str]) -> list[int]:
        return [self.indice_obs[self.observacion(w)] for w in palabras]

    # ------------------------------------------------------------------ inferencia

    def logprob_conjunta(self, palabras: list[str], etiquetas: list[str]) -> float:
        """log P(palabras, etiquetas) bajo el modelo generativo (AIMA §24.1.6, pp. 880–882).

        log π(e_1) + log B_{e_1}(w_1) + Σ_{k≥2} [log A(e_{k-1}, e_k) + log B_{e_k}(w_k)] (+ log fin(e_n)).
        Para la oración vacía devuelve 0.0.
        """
        self._exigir_entrenado()
        if len(palabras) != len(etiquetas):
            raise ValueError(f"{len(palabras)} palabras y {len(etiquetas)} etiquetas")
        if not palabras:
            return 0.0
        desconocidas = [e for e in etiquetas if e not in self.indice_etiqueta]
        if desconocidas:
            raise ValueError(f"etiquetas que el modelo no conoce: {desconocidas}")
        obs = self._indices_obs(palabras)
        est = [self.indice_etiqueta[e] for e in etiquetas]
        total = float(self.log_pi[est[0]]) + float(self.log_B[est[0], obs[0]])
        for k in range(1, len(est)):
            total += float(self.log_A[est[k - 1], est[k]]) + float(self.log_B[est[k], obs[k]])
        if self.usar_fin:
            total += float(self.log_fin[est[-1]])
        return total

    def viterbi(self, palabras: list[str]) -> tuple[list[str], float]:
        """Secuencia de etiquetas más probable y su log-probabilidad conjunta con las palabras.

        Algoritmo de Viterbi, AIMA §24.1.6, p. 881 (que remite el algoritmo a §14.2.3): programación
        dinámica en log.
        Devuelve ([], 0.0) para la oración vacía.
        """
        return self._decodificar(palabras, {})

    def tabla_viterbi(self, palabras: list[str]) -> np.ndarray:
        """Copia de la tabla δ (n × T) de Viterbi para las palabras; columnas en el orden de `etiquetas`.

        δ[k, j] = max sobre e_1..e_{k-1} de log P(w_1..w_k, e_1..e_{k-1}, e_k = j). No incluye FIN.
        """
        self._decodificar(palabras, {})
        return self.delta.copy()

    def _decodificar(
        self, palabras: list[str], restricciones: dict[int, set[str]]
    ) -> tuple[list[str], float]:
        """Viterbi con tablas NumPy δ (float, n × T) y ψ (int, n × T) y retroceso (AIMA §24.1.6, p. 881,
        que remite el algoritmo a §14.2.3).

        restricciones: {posición: etiquetas permitidas}; las demás reciben -inf en esa posición.
            δ[0, j]  = log π_j + log B_j(w_0)
            δ[k, j]  = max_i (δ[k-1, i] + log A_ij) + log B_j(w_k),   ψ[k, j] = argmax_i (...)
            final_j  = δ[n-1, j] (+ log fin_j)
        Empates: argmax de NumPy toma el primer máximo (orden de `etiquetas`), así que es determinista.
        """
        self._exigir_entrenado()
        n, T = len(palabras), len(self.etiquetas)
        self.delta = np.zeros((n, T))
        self.psi = np.zeros((n, T), dtype=int)
        if n == 0:
            return [], 0.0

        mascara = np.zeros((n, T))
        for pos, permitidas in restricciones.items():
            for t, k in self.indice_etiqueta.items():
                if t not in permitidas:
                    mascara[pos, k] = -np.inf

        obs = self._indices_obs(palabras)
        delta, psi = self.delta, self.psi
        delta[0] = self.log_pi + self.log_B[:, obs[0]] + mascara[0]
        for k in range(1, n):
            candidatos = delta[k - 1][:, None] + self.log_A  # candidatos[i, j]: venir de i e ir a j
            psi[k] = candidatos.argmax(axis=0)
            delta[k] = candidatos.max(axis=0) + self.log_B[:, obs[k]] + mascara[k]

        final = delta[n - 1] + self.log_fin if self.usar_fin else delta[n - 1]
        mejor = int(final.argmax())
        logprob = float(final[mejor])
        if logprob == -np.inf:  # defensa: con α > 0 solo ocurriría si la máscara lo excluyera todo
            raise ValueError("ninguna secuencia de etiquetas tiene probabilidad > 0 con estas restricciones")
        camino = [mejor]
        for k in range(n - 1, 0, -1):
            camino.append(int(psi[k, camino[-1]]))
        camino.reverse()
        return [self.etiquetas[j] for j in camino], logprob

    def etiquetas_unk(
        self, palabras: list[str], posiciones: list[int], permitidas: set[str] | None = None
    ) -> dict[int, str]:
        """Etiqueta propuesta para cada posición pedida, en el formato de `ParserCYK.analizar(etiquetas_unk=...)`.

        Se decodifica la **oración completa** con Viterbi (AIMA §24.1.6, p. 881): el contexto de las palabras
        conocidas es lo que permite acertar la etiqueta de la desconocida. Si se da `permitidas` (p. ej. los
        preterminales de la PCFG), en las posiciones pedidas las demás etiquetas se excluyen (-inf) antes de
        decodificar, para que el CYK nunca reciba una categoría sin reglas.
        """
        self._exigir_entrenado()
        n = len(palabras)
        fuera = [p for p in posiciones if not (0 <= p < n)]
        if fuera:
            raise ValueError(f"posiciones fuera de rango para {n} palabras: {fuera}")
        restricciones: dict[int, set[str]] = {}
        if permitidas is not None:
            if not set(permitidas) & set(self.etiquetas):
                raise ValueError("ninguna etiqueta del modelo está entre las permitidas")
            restricciones = {p: set(permitidas) for p in posiciones}
        etiquetas, _ = self._decodificar(palabras, restricciones)
        return {p: etiquetas[p] for p in posiciones}

    def _exigir_entrenado(self) -> None:
        if not self._entrenado:
            raise RuntimeError("el HMM no está entrenado: llame primero a entrenar() o use HMM.cargar()")

    # ------------------------------------------------------------------ persistencia (C6)

    def guardar(self, ruta: str | Path) -> None:
        """JSON UTF-8 con hiperparámetros y conteos enteros (las tablas de log se recalculan al cargar)."""
        self._exigir_entrenado()
        datos = {
            "tipo": self.TIPO,
            "version": VERSION,
            "alfa_trans": self.alfa_trans,
            "alfa_emis": self.alfa_emis,
            "umbral_raras": self.umbral_raras,
            "usar_fin": self.usar_fin,
            "etiquetas": self.etiquetas,
            "n_oraciones": self.n_oraciones,
            "conteo_inicio": self.conteo_inicio,
            "conteo_trans": self.conteo_trans,
            "conteo_fin": self.conteo_fin,
            "conteo_etiqueta": self.conteo_etiqueta,
            "conteo_emision": {t: dict(sorted(c.items())) for t, c in self.conteo_emision.items()},
            "conocidas": sorted(self.conocidas),
            "firmas": sorted(self.firmas),
        }
        escribir_json(ruta, datos)

    @classmethod
    def cargar(cls, ruta: str | Path) -> HMM:
        """Lee un HMM guardado con `guardar`; valida "tipo", "version" y los campos, y recalcula las tablas.

        ValueError si el JSON no es un objeto, le faltan campos o los conteos no tienen la forma esperada.
        """
        datos = leer_json(ruta, cls.TIPO, cls.CAMPOS)
        try:
            modelo = cls(
                alfa_trans=datos["alfa_trans"],
                alfa_emis=datos["alfa_emis"],
                umbral_raras=datos["umbral_raras"],
                usar_fin=datos["usar_fin"],
            )
            modelo.etiquetas = list(datos["etiquetas"])
            modelo.n_oraciones = int(datos["n_oraciones"])
            modelo.conteo_inicio = {t: int(c) for t, c in datos["conteo_inicio"].items()}
            modelo.conteo_trans = {
                i: {j: int(c) for j, c in fila.items()} for i, fila in datos["conteo_trans"].items()
            }
            modelo.conteo_fin = {t: int(c) for t, c in datos["conteo_fin"].items()}
            modelo.conteo_etiqueta = {t: int(c) for t, c in datos["conteo_etiqueta"].items()}
            modelo.conteo_emision = {
                t: {w: int(c) for w, c in fila.items()} for t, fila in datos["conteo_emision"].items()
            }
            modelo.conocidas = set(datos["conocidas"])
            modelo.firmas = set(datos["firmas"])
            modelo._calcular_tablas()
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"el archivo del HMM está mal formado: {e!r}") from None
        return modelo


__all__ = ["HMM", "firma", "SUFIJOS", "TERMINALES_ESPECIALES", "FIRMA_DESCONOCIDA", "FIRMA_DIGITO"]
