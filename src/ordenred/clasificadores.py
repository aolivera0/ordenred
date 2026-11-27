"""M3 · Clasificación de la intención (contrato C6). Dueño: rol 4.

Dos clasificadores de texto sobre los terminales de la orden (`Token.terminal`: "bloquea", "<CIDR>"…);
las clases son las acciones de FormaLogica (`semantica.ACCIONES`).

- `NaiveBayes`: modelo **generativo** multinomial con bolsa de palabras, AIMA 4.ª ed. global,
  §24.1.1, pp. 875–876, con suavizado de Laplace (aditivo), §24.1.4, p. 878.
- `RegresionLogistica`: modelo **discriminativo** multiclase (*softmax*) entrenado por descenso por
  gradiente con regularización L2, §24.1.6, pp. 881–883 (allí se comparan los enfoques generativo y
  discriminativo). Para el modelo en sí el libro remite a §19.6.5, p. 702. En §24.1.6 el libro usa un
  modelo binario por etiqueta (uno por cada categoría); aquí se usa la versión *softmax* multiclase.

Todas las probabilidades se manejan en logaritmo y se normalizan al final con log-sum-exp
(`_normalizar_log`). Persistencia en JSON (`guardar` / `cargar`) según los acuerdos del rol 4 en C6.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import numpy as np

from ._persistencia import VERSION, escribir_json, leer_json

# Diferencia mínima entre log-puntajes para no considerarlos empatados (ver _argmax_determinista).
TOLERANCIA_EMPATE = 1e-9


class Clasificador(Protocol):
    def entrenar(self, X: list[list[str]], y: list[str]) -> None: ...

    def predecir(self, tokens: list[str]) -> tuple[str, dict[str, float]]:
        """Devuelve (clase más probable, {clase: P(clase | tokens)}) con probabilidades que suman 1."""
        ...


# --------------------------------------------------------------------------------------------
# Utilidades comunes
# --------------------------------------------------------------------------------------------


def _normalizar_log(logs: dict[str, float]) -> dict[str, float]:
    """Convierte puntajes logarítmicos no normalizados en probabilidades que suman 1.

    log-sum-exp a mano: P(c) = exp(l_c − m) / Σ_k exp(l_k − m), con m = max_k l_k. Restar el máximo
    evita el *underflow* de exp() cuando los puntajes son muy negativos.
    """
    if not logs:
        raise ValueError("no hay clases que normalizar")
    m = max(logs.values())
    if m == -math.inf:
        raise ValueError("todas las clases tienen probabilidad 0")
    exps = {c: math.exp(v - m) for c, v in logs.items()}
    z = math.fsum(exps.values())
    return {c: e / z for c, e in exps.items()}


def _argmax_determinista(clases: list[str], puntajes: dict[str, float]) -> str:
    """Clase de mayor puntaje logarítmico; en empate, la primera de `clases` (que va ordenada).

    Dos puntajes que son iguales en aritmética exacta pueden diferir en el último bit al sumar
    logaritmos en coma flotante (p. ej. log(4/9) + log(3/6) frente a log(5/9) + log(2/5), ambos
    log(2/9)). Por eso solo se cambia de mejor si la ventaja supera TOLERANCIA_EMPATE; así el
    desempate por orden alfabético se respeta también en esos casos.

    La tolerancia es **absoluta** (1e-9 en escala logarítmica): equivale a un cociente de
    probabilidades de 1 + 1e-9, así que solo afecta a puntajes que en la práctica están empatados.

    Con −inf: (−inf) − (−inf) es nan y `nan > tolerancia` es falso, de modo que nunca cambia el
    mejor; un puntaje finito frente a −inf da +inf y sí lo cambia.
    """
    mejor = clases[0]
    for c in clases[1:]:
        if puntajes[c] - puntajes[mejor] > TOLERANCIA_EMPATE:
            mejor = c
    return mejor


def _validar_datos(X: list[list[str]], y: list[str]) -> None:
    if len(X) == 0:
        raise ValueError("no hay documentos de entrenamiento")
    if len(X) != len(y):
        raise ValueError(f"X tiene {len(X)} documentos pero y tiene {len(y)} etiquetas")


# --------------------------------------------------------------------------------------------
# Naive Bayes multinomial
# --------------------------------------------------------------------------------------------


class NaiveBayes:
    """Naive Bayes multinomial con bolsa de palabras, AIMA §24.1.1, pp. 875–876.

    Modelo generativo: P(c | w_1..w_n) ∝ P(c) · Π_i P(w_i | c), suponiendo las palabras independientes
    dada la clase y sin importar su orden (bolsa de palabras).

    Estimación por conteo con suavizado de Laplace (aditivo), AIMA §24.1.4, p. 878:

        log P(c)     = log(N_c / N)
        log P(w | c) = log((n_{c,w} + α) / (n_c + α·|V|))

    donde N_c es el número de documentos de la clase c, N el total de documentos, n_{c,w} las veces
    que aparece w en los documentos de c (con repeticiones), n_c = Σ_w n_{c,w} y V el vocabulario
    (unión de las palabras de entrenamiento).

    Palabras fuera de V: se **ignoran** al predecir (no aportan evidencia a ninguna clase). Por eso
    una orden hecha solo de palabras desconocidas, o vacía, recibe exactamente el prior P(c).

    Con alfa = 0 (sin suavizado) una palabra no vista en una clase le da probabilidad 0 (log = −inf);
    si eso ocurre en todas las clases a la vez, `predecir` lanza ValueError.
    """

    TIPO = "NaiveBayes"

    def __init__(self, alfa: float = 1.0) -> None:
        if not (0 <= alfa < math.inf):  # también rechaza nan
            raise ValueError(f"alfa debe ser >= 0 y finito, no {alfa}")
        self.alfa = alfa  # suavizado de Laplace (aditivo)
        self.clases: list[str] = []
        self.vocabulario: list[str] = []
        self.docs_por_clase: dict[str, int] = {}  # N_c
        self.conteo_palabras: dict[str, dict[str, int]] = {}  # n_{c,w} (solo las no nulas)
        self.palabras_por_clase: dict[str, int] = {}  # n_c
        self.log_prior: dict[str, float] = {}
        self.log_verosimilitud: dict[str, dict[str, float]] = {}
        self._entrenado = False

    # -- entrenamiento ---------------------------------------------------------------------

    def entrenar(self, X: list[list[str]], y: list[str]) -> None:
        """Cuenta N_c, n_{c,w} y n_c, y calcula las log-probabilidades suavizadas."""
        _validar_datos(X, y)
        docs: dict[str, int] = {}
        conteo: dict[str, dict[str, int]] = {}
        for tokens, c in zip(X, y):
            docs[c] = docs.get(c, 0) + 1
            fila = conteo.setdefault(c, {})
            for w in tokens:
                fila[w] = fila.get(w, 0) + 1
        self._fijar_conteos(docs, conteo)

    def _fijar_conteos(self, docs: dict[str, int], conteo: dict[str, dict[str, int]]) -> None:
        self.clases = sorted(docs)
        self.docs_por_clase = {c: docs[c] for c in self.clases}
        self.conteo_palabras = {c: dict(sorted(conteo.get(c, {}).items())) for c in self.clases}
        self.palabras_por_clase = {c: sum(self.conteo_palabras[c].values()) for c in self.clases}
        self.vocabulario = sorted({w for fila in self.conteo_palabras.values() for w in fila})
        self._recalcular_logs()
        self._entrenado = True

    def _recalcular_logs(self) -> None:
        """Log-probabilidades a partir de los conteos enteros (también se usa al cargar)."""
        n_total = sum(self.docs_por_clase.values())
        tam_v = len(self.vocabulario)
        self.log_prior = {c: math.log(self.docs_por_clase[c] / n_total) for c in self.clases}
        self.log_verosimilitud = {}
        for c in self.clases:
            fila = self.conteo_palabras[c]
            denominador = self.palabras_por_clase[c] + self.alfa * tam_v
            logs = {}
            for w in self.vocabulario:
                numerador = fila.get(w, 0) + self.alfa
                # Con alfa = 0 una palabra no vista en c tiene probabilidad 0 (log = −inf).
                logs[w] = math.log(numerador / denominador) if numerador > 0 else -math.inf
            self.log_verosimilitud[c] = logs

    # -- predicción -------------------------------------------------------------------------

    def puntajes_log(self, tokens: list[str]) -> dict[str, float]:
        """log P(c) + Σ_i log P(w_i | c) sobre los tokens que están en V (sin normalizar)."""
        if not self._entrenado:
            raise RuntimeError("NaiveBayes no está entrenado: llame a entrenar() o cargar()")
        puntajes = {}
        for c in self.clases:
            lv = self.log_verosimilitud[c]
            s = self.log_prior[c]
            for w in tokens:
                if w in lv:  # las palabras fuera de V se ignoran
                    s += lv[w]
            puntajes[c] = s
        return puntajes

    def predecir(self, tokens: list[str]) -> tuple[str, dict[str, float]]:
        """(clase más probable, {clase: P(clase | tokens)}). En empate gana la primera en orden alfabético."""
        puntajes = self.puntajes_log(tokens)
        if len(self.clases) == 1:
            return self.clases[0], {self.clases[0]: 1.0}
        return _argmax_determinista(self.clases, puntajes), _normalizar_log(puntajes)

    def tabla_conteos(self) -> dict:
        """Tabla de conteos para hacer el cálculo a mano (ejercicio de la sustentación).

        Claves: "N" (documentos), "N_c", "n_c", "n_cw" (tabla completa clase × palabra, con ceros),
        "V" (tamaño del vocabulario), "vocabulario" y "alfa".
        """
        if not self._entrenado:
            raise RuntimeError("NaiveBayes no está entrenado: llame a entrenar() o cargar()")
        return {
            "N": sum(self.docs_por_clase.values()),
            "N_c": dict(self.docs_por_clase),
            "n_c": dict(self.palabras_por_clase),
            "n_cw": {
                c: {w: self.conteo_palabras[c].get(w, 0) for w in self.vocabulario} for c in self.clases
            },
            "V": len(self.vocabulario),
            "vocabulario": list(self.vocabulario),
            "alfa": self.alfa,
        }

    # -- persistencia ------------------------------------------------------------------------

    def guardar(self, ruta: str | Path) -> None:
        """JSON con conteos enteros; los logaritmos se recalculan al cargar (ida y vuelta exacta)."""
        if not self._entrenado:
            raise RuntimeError("NaiveBayes no está entrenado: no hay nada que guardar")
        escribir_json(
            ruta,
            {
                "tipo": self.TIPO,
                "version": VERSION,
                "alfa": self.alfa,
                "clases": self.clases,
                "docs_por_clase": self.docs_por_clase,
                "conteo_palabras": self.conteo_palabras,
            },
        )

    @classmethod
    def cargar(cls, ruta: str | Path) -> "NaiveBayes":
        """Lee un modelo guardado con `guardar`. ValueError si el JSON no tiene la forma esperada."""
        datos = leer_json(ruta, cls.TIPO, ("alfa", "docs_por_clase", "conteo_palabras"))
        try:
            modelo = cls(alfa=datos["alfa"])
            docs = {c: int(n) for c, n in datos["docs_por_clase"].items()}
            conteo = {c: {w: int(k) for w, k in fila.items()} for c, fila in datos["conteo_palabras"].items()}
            modelo._fijar_conteos(docs, conteo)
        except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            raise ValueError(f"el archivo de NaiveBayes está mal formado: {e!r}") from None
        return modelo


# --------------------------------------------------------------------------------------------
# Regresión logística multiclase (softmax)
# --------------------------------------------------------------------------------------------


class RegresionLogistica:
    """Regresión logística multiclase (*softmax*), AIMA §24.1.6, pp. 881–883 (que remite a §19.6.5, p. 702).

    El libro entrena un modelo binario por etiqueta; aquí un solo modelo *softmax* reparte la probabilidad
    entre las K clases.

    Modelo discriminativo: estima directamente P(c | x) = softmax(x·W)_c sin modelar cómo se
    generan las palabras (a diferencia de Naive Bayes, que es generativo).

    - Vectorizador propio: vocabulario ordenado → índice; cada documento es una fila con el conteo de
      cada palabra y una última columna de sesgo igual a 1.0. Las palabras fuera del vocabulario
      aportan 0.
    - W tiene forma (|V| + 1) × K; la última fila es el sesgo.
    - Pérdida: entropía cruzada media más L2 (sin regularizar el sesgo),
          L = −(1/N) Σ_i log softmax(X W)_{i, y_i} + (λ/2)·‖W_{sin sesgo}‖²
      con gradiente ∇ = (1/N) Xᵀ(P − Y) + λ·W (fila del sesgo sin término L2).
    - Descenso por gradiente por lotes completos durante `epocas`, con W inicial
      `default_rng(semilla).normal(0, 0.01)`: el entrenamiento es determinista.
    """

    TIPO = "RegresionLogistica"

    def __init__(self, tasa: float = 0.1, epocas: int = 200, l2: float = 1e-3, semilla: int = 2026) -> None:
        if not (0 < tasa < math.inf):
            raise ValueError(f"tasa debe ser > 0 y finita, no {tasa}")
        if epocas < 0:
            raise ValueError(f"epocas debe ser >= 0, no {epocas}")
        if not (0 <= l2 < math.inf):
            raise ValueError(f"l2 debe ser >= 0 y finito, no {l2}")
        self.tasa = tasa
        self.epocas = epocas
        self.l2 = l2
        self.semilla = semilla
        self.clases: list[str] = []
        self.vocabulario: list[str] = []
        self.indice: dict[str, int] = {}
        self.W: np.ndarray | None = None
        self.historial_perdida: list[float] = []
        self._entrenado = False

    # -- vectorización -----------------------------------------------------------------------

    def _fijar_vocabulario(self, vocabulario: list[str]) -> None:
        self.vocabulario = sorted(vocabulario)
        self.indice = {w: i for i, w in enumerate(self.vocabulario)}

    def vectorizar(self, X: list[list[str]]) -> np.ndarray:
        """Matriz N × (|V| + 1): conteos por palabra y sesgo 1.0 en la última columna."""
        d = len(self.vocabulario)
        M = np.zeros((len(X), d + 1))
        for i, tokens in enumerate(X):
            for w in tokens:
                j = self.indice.get(w)
                if j is not None:  # fuera del vocabulario: aporta 0
                    M[i, j] += 1.0
            M[i, d] = 1.0
        return M

    # -- pérdida y gradiente -----------------------------------------------------------------

    def _perdida_y_gradiente(self, W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
        """Entropía cruzada media + L2 y su gradiente respecto de W (Y en one-hot, N × K)."""
        n = X.shape[0]
        Z = X @ W
        Z = Z - Z.max(axis=1, keepdims=True)  # softmax estable
        log_z = np.log(np.exp(Z).sum(axis=1, keepdims=True))
        log_p = Z - log_z
        P = np.exp(log_p)
        sin_sesgo = W[:-1]
        perdida = -(log_p * Y).sum() / n + 0.5 * self.l2 * (sin_sesgo * sin_sesgo).sum()
        gradiente = (X.T @ (P - Y)) / n
        gradiente[:-1] = gradiente[:-1] + self.l2 * sin_sesgo
        return float(perdida), gradiente

    # -- entrenamiento -----------------------------------------------------------------------

    def entrenar(self, X: list[list[str]], y: list[str]) -> None:
        """Descenso por gradiente por lotes completos, AIMA §24.1.6, pp. 881–883.

        Durante `epocas` iteraciones: W ← W − tasa·∇L(W) sobre todos los documentos a la vez.
        `historial_perdida[t]` es la pérdida L(W) antes de la actualización t.
        """
        _validar_datos(X, y)
        self.clases = sorted(set(y))
        self._fijar_vocabulario(list({w for tokens in X for w in tokens}))
        self.historial_perdida = []
        k = len(self.clases)
        if k == 1:  # una sola clase: no hay nada que discriminar
            self.W = None
            self._entrenado = True
            return
        M = self.vectorizar(X)
        Y = np.zeros((len(y), k))
        posicion = {c: j for j, c in enumerate(self.clases)}
        for i, c in enumerate(y):
            Y[i, posicion[c]] = 1.0
        rng = np.random.default_rng(self.semilla)
        W = rng.normal(0.0, 0.01, size=(len(self.vocabulario) + 1, k))
        for _ in range(self.epocas):
            perdida, gradiente = self._perdida_y_gradiente(W, M, Y)
            self.historial_perdida.append(perdida)
            W = W - self.tasa * gradiente
        self.W = W
        self._entrenado = True

    # -- predicción --------------------------------------------------------------------------

    def predecir(self, tokens: list[str]) -> tuple[str, dict[str, float]]:
        """(clase más probable, softmax de x·W). En empate gana la primera en orden alfabético."""
        if not self._entrenado:
            raise RuntimeError("RegresionLogistica no está entrenada: llame a entrenar() o cargar()")
        if len(self.clases) == 1:
            return self.clases[0], {self.clases[0]: 1.0}
        z = self.vectorizar([tokens])[0] @ self.W
        puntajes = {c: float(z[j]) for j, c in enumerate(self.clases)}
        return _argmax_determinista(self.clases, puntajes), _normalizar_log(puntajes)

    # -- persistencia ------------------------------------------------------------------------

    def guardar(self, ruta: str | Path) -> None:
        """JSON con hiperparámetros, vocabulario, clases y W (listas de floats, ida y vuelta exacta)."""
        if not self._entrenado:
            raise RuntimeError("RegresionLogistica no está entrenada: no hay nada que guardar")
        escribir_json(
            ruta,
            {
                "tipo": self.TIPO,
                "version": VERSION,
                "tasa": self.tasa,
                "epocas": self.epocas,
                "l2": self.l2,
                "semilla": self.semilla,
                "clases": self.clases,
                "vocabulario": self.vocabulario,
                "W": None if self.W is None else self.W.tolist(),
                "historial_perdida": self.historial_perdida,
            },
        )

    @classmethod
    def cargar(cls, ruta: str | Path) -> "RegresionLogistica":
        """Lee un modelo guardado con `guardar`. ValueError si el JSON no tiene la forma esperada."""
        datos = leer_json(ruta, cls.TIPO, ("tasa", "epocas", "l2", "semilla", "clases", "vocabulario", "W"))
        try:
            modelo = cls(tasa=datos["tasa"], epocas=datos["epocas"], l2=datos["l2"], semilla=datos["semilla"])
            modelo.clases = list(datos["clases"])
            modelo._fijar_vocabulario(list(datos["vocabulario"]))
            modelo.W = None if datos["W"] is None else np.array(datos["W"], dtype=float)
            modelo.historial_perdida = [float(v) for v in datos.get("historial_perdida", [])]
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"el archivo de RegresionLogistica está mal formado: {e!r}") from None
        if modelo.W is None and len(modelo.clases) > 1:
            raise ValueError("W es null pero hay más de una clase: el modelo no tiene pesos")
        if modelo.W is not None and modelo.W.shape != (len(modelo.vocabulario) + 1, len(modelo.clases)):
            raise ValueError(f"W tiene forma {modelo.W.shape}, incompatible con vocabulario y clases")
        modelo._entrenado = True
        return modelo
