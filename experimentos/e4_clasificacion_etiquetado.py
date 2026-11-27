"""Escenario E4 · Tamaño del entrenamiento y palabras desconocidas. Dueño del código: rol 4.

Qué mide
--------
1. **Curvas de aprendizaje** de la clasificación de la intención: Naive Bayes (generativo, AIMA 4.ª ed.
   global, §24.1.1, pp. 875–876) frente a regresión logística (discriminativa, §24.1.6, pp. 881–883),
   entrenados con el 10, 25, 50 y 100 % de `train` y evaluados sobre el conjunto de evaluación
   (`dev` por omisión). Métrica: exactitud de la intención.
2. **Etiquetado de palabras desconocidas con el HMM** (Viterbi, §24.1.6, pp. 880–881): exactitud global,
   en palabras conocidas y en desconocidas, frente a dos líneas base «sin HMM» sin contexto: (a) la
   etiqueta más frecuente entre las palabras raras del entrenamiento y (b) la etiqueta más frecuente para
   la firma de la palabra (sufijo, dígitos). Como el corpus sintético tiene vocabulario cerrado, las
   desconocidas se crean artificialmente con `retener_vocabulario` (condición "retenido", que además
   informa la exactitud solo sobre las palabras retenidas); la condición "completo" usa `train` entero,
   donde las desconocidas son sobre todo las palabras raras que el HMM entrena como su firma.
3. **Matriz de confusión** de intenciones (las 7 acciones de `semantica.ACCIONES`) de NB y RL entrenados
   con todo `train`. Esta tabla es idéntica en todas las semillas (NB es determinista y la RL usa su
   semilla interna), así que su intervalo de confianza tiene ancho 0: basta reportar una semilla.

Hipótesis (breve)
-----------------
- Con pocos datos Naive Bayes iguala o supera a la regresión logística (el modelo generativo converge
  antes); con todo `train` la RL lo alcanza o lo supera (§24.1.6, discusión generativo/discriminativo).
- En las palabras desconocidas: base (a) < base (b) < HMM. La firma sola ya aporta (b − a); lo que el HMM
  gana sobre (b) es el aporte del contexto (las transiciones entre etiquetas).

Cómo correrlo (desde la raíz del repositorio)
---------------------------------------------
    docker compose run --rm ordenred python experimentos/e4_clasificacion_etiquetado.py --semillas 5
    docker compose run --rm ordenred python experimentos/e4_clasificacion_etiquetado.py --fixtures --salida /tmp/e4

Sin Docker hace falta `PYTHONPATH=src`. Escribe `e4_curvas.csv`, `e4_hmm_unk.csv` y `e4_confusion.csv` en
`--salida` (por omisión `experimentos/resultados/`). No dibuja figuras: las hace el rol 7 con estos CSV.
`correr(semilla, rutas, modo)` devuelve filas planas con la columna "tabla", lista para
`runner.correr("E4", semillas)`.

Entrada de los modelos: los **terminales** (`Token.terminal`), según los acuerdos del rol 4 en C6. En modo
"tokenizar" salen de `tokenizador.tokenizar` (rol 5); en modo "fixtures" de `tests.tokens_fixture`, que
solo sirve para `tests/fixtures/mini_corpus.jsonl` (5 órdenes, 2 clases: los números no significan nada).
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

from ordenred.clasificadores import NaiveBayes, RegresionLogistica
from ordenred.corpus import cargar_jsonl
from ordenred.hmm import HMM, TERMINALES_ESPECIALES
from ordenred.semantica import ACCIONES

RAIZ = Path(__file__).resolve().parent.parent
TRAIN_POR_OMISION = RAIZ / "datos" / "train.jsonl"
EVAL_POR_OMISION = RAIZ / "datos" / "dev.jsonl"
FIXTURE_CORPUS = RAIZ / "tests" / "fixtures" / "mini_corpus.jsonl"
SALIDA_POR_OMISION = RAIZ / "experimentos" / "resultados"

FRACCIONES = (0.1, 0.25, 0.5, 1.0)
PROPORCION_RETENIDA = 0.1
# Una palabra que aparece en más de esta fracción de las oraciones de train no se retiene (quitaría demasiadas).
MAX_FRACCION_ORACIONES = 0.3
# Intentos de muestreo de `retener_vocabulario` hasta conservar todas las etiquetas de train.
MAX_INTENTOS_RETENCION = 100
MODOS = ("tokenizar", "fixtures")
TABLAS = ("curvas", "hmm_unk", "confusion")
ARCHIVOS = {t: f"e4_{t}.csv" for t in TABLAS}


# --------------------------------------------------------------------------------------------
# Entrada
# --------------------------------------------------------------------------------------------


def terminales_registro(r: dict, modo: str) -> list[str]:
    """Lista de `Token.terminal` de un registro del corpus (C5).

    modo="fixtures": `tests.tokens_fixture(r["tokens"])` (solo distingue IP, CIDR y NUM; vale para los
    fixtures). modo="tokenizar": `tokenizador.tokenizar(r["orden"])`; las formas deben coincidir con
    `r["tokens"]`, si no ValueError con el id. Si `tokenizar` aún es un stub (NotImplementedError) se lanza
    RuntimeError: no hay respaldo silencioso.
    """
    if modo == "fixtures":
        if str(RAIZ) not in sys.path:
            sys.path.insert(0, str(RAIZ))
        from tests import tokens_fixture

        return [t.terminal for t in tokens_fixture(r["tokens"])]
    if modo == "tokenizar":
        from ordenred.tokenizador import tokenizar

        try:
            tokens = tokenizar(r["orden"])
        except NotImplementedError as e:
            raise RuntimeError("el tokenizador del rol 5 aún no existe; use --fixtures") from e
        formas = [t.forma for t in tokens]
        if formas != r["tokens"]:
            raise ValueError(
                f"registro {r.get('id')!r}: tokenizar da {formas} pero el corpus trae {r['tokens']}"
            )
        return [t.terminal for t in tokens]
    raise ValueError(f"modo desconocido: {modo!r}; válidos: {MODOS}")


def cargar_corpus(ruta: str | Path) -> list[dict]:
    """`corpus.cargar_jsonl` con un error claro si el archivo no existe (el corpus es del rol 5)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"no existe {ruta}: el corpus (datos/*.jsonl) lo genera el rol 5; "
            "para probar el flujo de punta a punta use --fixtures"
        )
    return cargar_jsonl(ruta)


def submuestra(registros: list[dict], fraccion: float, semilla: int) -> list[dict]:
    """max(1, round(fraccion·N)) registros elegidos sin reemplazo con `random.Random(semilla)`.

    Determinista: la misma semilla da la misma submuestra. Conserva el orden original del corpus.
    `round` es el de Python (redondeo al par: round(0.5) = 0, de ahí el max con 1).
    """
    if not registros:
        raise ValueError("no hay registros que submuestrear")
    if not (0 < fraccion <= 1):
        raise ValueError(f"la fracción debe estar en (0, 1], no {fraccion}")
    k = max(1, round(fraccion * len(registros)))
    indices = sorted(random.Random(semilla).sample(range(len(registros)), k))
    return [registros[i] for i in indices]


# --------------------------------------------------------------------------------------------
# Clasificación de la intención (NB vs. RL)
# --------------------------------------------------------------------------------------------


def _nuevo_clasificador(modelo: str):
    if modelo == "nb":
        return NaiveBayes()
    if modelo == "rl":
        return RegresionLogistica()  # hiperparámetros por omisión (semilla interna fija)
    raise ValueError(f"modelo desconocido: {modelo!r}")


def _entrenar_clasificador(modelo: str, registros: list[dict], modo: str):
    clf = _nuevo_clasificador(modelo)
    clf.entrenar([terminales_registro(r, modo) for r in registros], [r["intencion"] for r in registros])
    return clf


def _exactitud(reales: list[str], predichas: list[str]) -> float:
    if not reales:
        return math.nan
    return sum(a == b for a, b in zip(reales, predichas)) / len(reales)


def curva_aprendizaje(
    train: list[dict], evaluacion: list[dict], fracciones, semillas, modo: str
) -> list[dict]:
    """Exactitud de intención de NB y RL entrenados con cada fracción de `train` y cada semilla.

    Para cada (semilla, fracción) se toma `submuestra(train, fraccion, semilla)` y con ella se entrenan
    ambos modelos (la misma submuestra para los dos, así la comparación es pareada). Filas:
    {"modelo": "nb" | "rl", "fraccion", "semilla", "n_train", "exactitud"}.
    """
    X_eval = [terminales_registro(r, modo) for r in evaluacion]
    y_eval = [r["intencion"] for r in evaluacion]
    filas = []
    for semilla in semillas:
        for fraccion in fracciones:
            sub = submuestra(train, fraccion, semilla)
            for modelo in ("nb", "rl"):
                clf = _entrenar_clasificador(modelo, sub, modo)
                predichas = [clf.predecir(x)[0] for x in X_eval]
                filas.append({
                    "modelo": modelo,
                    "fraccion": fraccion,
                    "semilla": semilla,
                    "n_train": len(sub),
                    "exactitud": _exactitud(y_eval, predichas),
                })
    return filas


def matriz_confusion(reales: list[str], predichas: list[str], clases=ACCIONES) -> dict[str, dict[str, int]]:
    """Matriz de confusión como dict anidado: m[real][predicha] = número de casos.

    Incluye todas las `clases` (por omisión las 7 de ACCIONES) aunque no aparezcan en los datos.
    Una etiqueta fuera de `clases` es un error (ValueError), no se descarta en silencio.
    """
    if len(reales) != len(predichas):
        raise ValueError(f"{len(reales)} etiquetas reales y {len(predichas)} predichas")
    clases = list(clases)
    m = {r: {p: 0 for p in clases} for r in clases}
    for r, p in zip(reales, predichas):
        if r not in m or p not in m:
            raise ValueError(f"etiqueta fuera de las clases: real={r!r}, predicha={p!r}")
        m[r][p] += 1
    return m


# --------------------------------------------------------------------------------------------
# HMM y palabras desconocidas
# --------------------------------------------------------------------------------------------


def _es_palabra(terminal: str) -> bool:
    """True para terminales de tipo PALABRA ("bloquea"); False para "<IP>", "<CIDR>", "<NUM>"…"""
    return terminal not in TERMINALES_ESPECIALES and not (terminal.startswith("<") and terminal.endswith(">"))


def retener_vocabulario(
    train: list[dict], proporcion: float, semilla: int, modo: str
) -> tuple[list[dict], set[str]]:
    """Crea palabras desconocidas artificiales: (train sin las oraciones que las contienen, retenidas).

    El corpus sintético tiene vocabulario cerrado, así que casi no hay palabras nuevas en `dev`. Se quitan
    de `train` todas las oraciones que contienen alguna palabra retenida, de modo que el modelo nunca las
    ve y en la evaluación son desconocidas de verdad. Para que la métrica mida el etiquetado de palabras
    nuevas y no la desaparición de etiquetas enteras:

    1. Solo son candidatas las palabras (terminales PALABRA; nunca "<IP>", "<CIDR>", "<NUM>"…) tales que
       (a) aparecen en registros con `etiquetas` de oro, (b) cada una de sus etiquetas de oro tiene al menos
       otro tipo de terminal en `train` (retener «puerto», único NPUERTO, borraría la etiqueta), y (c)
       aparecen en a lo sumo `MAX_FRACCION_ORACIONES` de las oraciones de `train`.
    2. k = round(proporcion · T), con T el número de tipos de palabra de `train` (al menos 1 si
       proporcion > 0, y a lo sumo el número de candidatas).
    3. Con un único `random.Random(semilla)` se muestrean k candidatas (ordenadas, así no depende del
       hash) hasta `MAX_INTENTOS_RETENCION` veces, y se acepta el primer intento cuyo `train` restante
       conserva exactamente el conjunto de etiquetas de `train` (una etiqueta puede depender de una
       oración, como NUMERO tras «puerto»). Determinista para cada semilla.

    proporcion = 0 devuelve `train` completo y un conjunto vacío. ValueError si no hay candidatas o si
    ningún intento conserva las etiquetas.
    """
    if not (0 <= proporcion <= 1):
        raise ValueError(f"la proporción debe estar en [0, 1], no {proporcion}")
    terminales = [terminales_registro(r, modo) for r in train]
    if proporcion == 0:
        return list(train), set()

    etiquetas_de: dict[str, set[str]] = {}  # palabra → etiquetas de oro con que aparece
    tipos_por_etiqueta: dict[str, set[str]] = {}  # etiqueta → terminales (de cualquier tipo) que la llevan
    for r, ts in zip(train, terminales):
        if r.get("etiquetas") is None:
            continue
        if len(ts) != len(r["etiquetas"]):
            raise ValueError(f"registro {r.get('id')!r}: {len(ts)} terminales y {len(r['etiquetas'])} etiquetas")
        for w, t in zip(ts, r["etiquetas"]):
            tipos_por_etiqueta.setdefault(t, set()).add(w)
            if _es_palabra(w):
                etiquetas_de.setdefault(w, set()).add(t)
    etiquetas_train = set(tipos_por_etiqueta)

    oraciones_con: dict[str, int] = {}
    for ts in terminales:
        for w in set(ts):
            oraciones_con[w] = oraciones_con.get(w, 0) + 1
    tipos = sorted(w for w in oraciones_con if _es_palabra(w))
    candidatos = sorted(
        w for w in etiquetas_de
        if all(len(tipos_por_etiqueta[t]) >= 2 for t in etiquetas_de[w])
        and oraciones_con[w] <= MAX_FRACCION_ORACIONES * len(train)
    )
    if not candidatos:
        raise ValueError("ninguna palabra de train se puede retener sin borrar una etiqueta entera")
    k = min(len(candidatos), max(1, round(proporcion * len(tipos))))

    rng = random.Random(semilla)
    for _ in range(MAX_INTENTOS_RETENCION):
        retenidas = set(rng.sample(candidatos, k))
        restantes = [r for r, ts in zip(train, terminales) if not retenidas.intersection(ts)]
        etiquetas_restantes = {t for r in _con_etiquetas(restantes) for t in r["etiquetas"]}
        if restantes and etiquetas_restantes == etiquetas_train:
            return restantes, retenidas
    raise ValueError(
        f"en {MAX_INTENTOS_RETENCION} intentos no se hallaron {k} palabras retenibles que conserven "
        "todas las etiquetas de train; baje la proporción"
    )


def _con_etiquetas(registros: list[dict]) -> list[dict]:
    """Registros con `etiquetas` (los humanos fuera de cobertura las tienen en null y se descartan)."""
    return [r for r in registros if r.get("etiquetas") is not None]


def entrenar_hmm(train: list[dict], modo: str) -> HMM:
    """HMM con sus hiperparámetros por omisión, entrenado con (terminal, etiqueta) de cada registro."""
    oraciones = []
    for r in _con_etiquetas(train):
        ts = terminales_registro(r, modo)
        if len(ts) != len(r["etiquetas"]):
            raise ValueError(f"registro {r.get('id')!r}: {len(ts)} terminales y {len(r['etiquetas'])} etiquetas")
        oraciones.append(list(zip(ts, r["etiquetas"])))
    hmm = HMM()
    hmm.entrenar(oraciones)
    return hmm


def _argmax_alfabetico(conteos: dict[str, int]) -> str:
    """Clave de mayor conteo; empate: la primera en orden alfabético."""
    return min(conteos, key=lambda t: (-conteos[t], t))


def etiqueta_base_raras(hmm: HMM) -> str:
    """Etiqueta más frecuente entre las palabras raras del entrenamiento del HMM.

    tot[t] = Σ conteo_emision[t][f] sobre las firmas f de `hmm.firmas` (las raras se entrenan como su
    firma); empate por orden alfabético.

    Respaldo si no hubo palabras raras (corpus de vocabulario cerrado, `n_raras_train` = 0): la etiqueta
    con más **tipos** de palabra conocidos (terminales PALABRA, no "<IP>"…), es decir, la de clase más
    abierta, que es donde suele caer una palabra nueva; empate alfabético. La etiqueta más frecuente global
    sería una de clase cerrada (DET) y haría la base trivial. Si ninguna etiqueta tiene palabras, la más
    frecuente global (`hmm.conteo_etiqueta`).
    """
    tot = {t: sum(c for w, c in hmm.conteo_emision[t].items() if w in hmm.firmas) for t in hmm.etiquetas}
    if any(tot.values()):
        return _argmax_alfabetico(tot)
    tipos = {
        t: sum(1 for w in hmm.conteo_emision[t] if w not in hmm.firmas and _es_palabra(w))
        for t in hmm.etiquetas
    }
    if any(tipos.values()):
        return _argmax_alfabetico(tipos)
    return _argmax_alfabetico({t: hmm.conteo_etiqueta[t] for t in hmm.etiquetas})


def n_raras_train(hmm: HMM) -> int:
    """Apariciones de palabras raras en el entrenamiento (emisiones contadas como firma). Si es 0, las
    bases de `exactitud_hmm` usan el respaldo y la de firma coincide con la de raras."""
    return sum(c for t in hmm.etiquetas for w, c in hmm.conteo_emision[t].items() if w in hmm.firmas)


def etiqueta_base_firma(hmm: HMM, palabra: str, respaldo: str) -> str:
    """Etiqueta más frecuente, sin contexto, para el símbolo con que el HMM ve la palabra.

    El símbolo es `hmm.observacion(palabra)` (la palabra, su firma o "<UNK>"; exactamente lo que usan las
    emisiones de Viterbi), así que HMM − esta base aísla el aporte del contexto. Se toma el argmax de
    `hmm.conteo_emision[t][símbolo]` (empate alfabético); si el símbolo no tiene conteos, `respaldo`
    (la etiqueta de `etiqueta_base_raras`).
    """
    simbolo = hmm.observacion(palabra)
    conteos = {t: hmm.conteo_emision[t].get(simbolo, 0) for t in hmm.etiquetas}
    return _argmax_alfabetico(conteos) if any(conteos.values()) else respaldo


def exactitud_hmm(hmm: HMM, evaluacion: list[dict], modo: str, retenidas=frozenset()) -> dict:
    """Exactitud de Viterbi por palabra: global, en conocidas, en desconocidas y en retenidas, y dos bases.

    - Se descartan los registros con `etiquetas` null (humanos fuera de cobertura).
    - Cada oración se decodifica **completa** con `hmm.viterbi` (el contexto ayuda a la desconocida).
    - Desconocida = `not hmm.es_conocida(w)`: palabra nunca vista o rara en el entrenamiento (el HMM
      entrena las raras como su firma), o un terminal especial que no se vio nunca. Mezcla varios casos;
      por eso `exactitud_retenidas` mide solo las apariciones de las palabras de `retenidas`
      (las de `retener_vocabulario`), y `n_retenidas_eval` cuántas son.
    - Líneas base «sin HMM», sobre las desconocidas, sin contexto:
      `exactitud_desconocidas_base`: la misma etiqueta para todas, `etiqueta_base_raras(hmm)` (la más
      frecuente entre las palabras raras del entrenamiento, o su respaldo; se informa en `etiqueta_base`).
      `exactitud_base_firma`: para cada una, `etiqueta_base_firma(hmm, w, etiqueta_base)` (la más
      frecuente para el símbolo `hmm.observacion(w)`). La diferencia HMM − base_firma es el aporte del
      contexto.
    - `n_raras_train`: si vale 0 (sin palabras raras en el entrenamiento), ambas bases usan el respaldo de
      `etiqueta_base_raras`, coinciden y no miden el aporte de la firma.

    Una exactitud sin casos (p. ej. ninguna desconocida) vale nan.
    """
    base = etiqueta_base_raras(hmm)
    n_oraciones = n = aciertos = 0
    n_con = aciertos_con = 0
    n_des = aciertos_des = aciertos_base = aciertos_firma = 0
    n_ret = aciertos_ret = 0
    for r in _con_etiquetas(evaluacion):
        ts = terminales_registro(r, modo)
        reales = r["etiquetas"]
        if len(ts) != len(reales):
            raise ValueError(f"registro {r.get('id')!r}: {len(ts)} terminales y {len(reales)} etiquetas")
        predichas, _ = hmm.viterbi(ts)
        n_oraciones += 1
        for w, real, pred in zip(ts, reales, predichas):
            acierto = real == pred
            n += 1
            aciertos += acierto
            if hmm.es_conocida(w):
                n_con += 1
                aciertos_con += acierto
            else:
                n_des += 1
                aciertos_des += acierto
                aciertos_base += real == base
                aciertos_firma += real == etiqueta_base_firma(hmm, w, base)
            if w in retenidas:
                n_ret += 1
                aciertos_ret += acierto
    return {
        "n_raras_train": n_raras_train(hmm),
        "n_oraciones": n_oraciones,
        "n_palabras": n,
        "exactitud": aciertos / n if n else math.nan,
        "n_conocidas": n_con,
        "exactitud_conocidas": aciertos_con / n_con if n_con else math.nan,
        "n_desconocidas": n_des,
        "exactitud_desconocidas": aciertos_des / n_des if n_des else math.nan,
        "etiqueta_base": base,
        "exactitud_desconocidas_base": aciertos_base / n_des if n_des else math.nan,
        "exactitud_base_firma": aciertos_firma / n_des if n_des else math.nan,
        "n_retenidas_eval": n_ret,
        "exactitud_retenidas": aciertos_ret / n_ret if n_ret else math.nan,
    }


# --------------------------------------------------------------------------------------------
# Escenario completo
# --------------------------------------------------------------------------------------------


def correr(
    semilla: int,
    rutas: dict,
    modo: str,
    fracciones=FRACCIONES,
    proporcion_retenida: float = PROPORCION_RETENIDA,
) -> list[dict]:
    """Corre E4 con una semilla y devuelve filas planas con la columna "tabla".

    rutas: {"train": ruta, "eval": ruta}. Tablas:
    - "curvas": `curva_aprendizaje` con esta semilla.
    - "hmm_unk": una fila por condición, "completo" (HMM con todo `train`) y "retenido" (HMM con `train`
      sin las oraciones de las palabras retenidas por `retener_vocabulario(..., semilla)`), con las
      métricas de `exactitud_hmm` (en "completo", `n_retenidas_eval` = 0 y `exactitud_retenidas` = nan).
    - "confusion": NB y RL con todo `train`; una fila por (modelo, real, predicha) con las 7 acciones.
      Con todo `train` NB es determinista y la RL usa su semilla interna, así que esta tabla es idéntica
      para toda `semilla` (intervalo de confianza de ancho 0).
    """
    train = cargar_corpus(rutas["train"])
    evaluacion = cargar_corpus(rutas["eval"])
    filas: list[dict] = []

    for f in curva_aprendizaje(train, evaluacion, fracciones, [semilla], modo):
        filas.append({"tabla": "curvas", **f})

    train_ret, retenidas = retener_vocabulario(train, proporcion_retenida, semilla, modo)
    condiciones = (
        ("completo", train, 0.0, set()),
        ("retenido", train_ret, proporcion_retenida, retenidas),
    )
    for condicion, datos, proporcion, ret in condiciones:
        hmm = entrenar_hmm(datos, modo)
        filas.append({
            "tabla": "hmm_unk",
            "semilla": semilla,
            "condicion": condicion,
            "proporcion_retenida": proporcion,
            "n_retenidas": len(ret),
            "n_train": len(datos),
            **exactitud_hmm(hmm, evaluacion, modo, ret),
        })

    X_eval = [terminales_registro(r, modo) for r in evaluacion]
    reales = [r["intencion"] for r in evaluacion]
    for modelo in ("nb", "rl"):
        clf = _entrenar_clasificador(modelo, train, modo)
        m = matriz_confusion(reales, [clf.predecir(x)[0] for x in X_eval], ACCIONES)
        for real in ACCIONES:
            for predicha in ACCIONES:
                filas.append({
                    "tabla": "confusion",
                    "semilla": semilla,
                    "modelo": modelo,
                    "real": real,
                    "predicha": predicha,
                    "n": m[real][predicha],
                })
    return filas


def escribir_csv(filas: list[dict], directorio: str | Path) -> dict[str, Path]:
    """Escribe una CSV (UTF-8) por tabla en `directorio`; columnas en orden de primera aparición.

    Una exactitud sin casos se escribe como el texto `nan`, que `float()` lee de vuelta como NaN.
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for tabla in TABLAS:
        de_tabla = [{k: v for k, v in f.items() if k != "tabla"} for f in filas if f["tabla"] == tabla]
        columnas: list[str] = []
        for f in de_tabla:
            columnas += [k for k in f if k not in columnas]
        ruta = directorio / ARCHIVOS[tabla]
        with open(ruta, "w", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=columnas)
            escritor.writeheader()
            escritor.writerows(de_tabla)
        rutas[tabla] = ruta
    return rutas


def _media(valores: list[float]) -> float:
    validos = [v for v in valores if not math.isnan(v)]
    return sum(validos) / len(validos) if validos else math.nan


def resumen(filas: list[dict]) -> str:
    """Texto breve: exactitud media por (modelo, fracción) y del HMM en desconocidas por condición."""
    lineas = ["Curvas de aprendizaje (exactitud media de intención):"]
    curvas = [f for f in filas if f["tabla"] == "curvas"]
    for fraccion in sorted({f["fraccion"] for f in curvas}):
        partes = []
        for modelo in ("nb", "rl"):
            vals = [f["exactitud"] for f in curvas if f["fraccion"] == fraccion and f["modelo"] == modelo]
            partes.append(f"{modelo} = {_media(vals):.3f}")
        lineas.append(f"  {fraccion:>5.0%}: " + ", ".join(partes))
    lineas.append("HMM en palabras desconocidas (media):")
    hmm = [f for f in filas if f["tabla"] == "hmm_unk"]
    for condicion in ("completo", "retenido"):
        de = [f for f in hmm if f["condicion"] == condicion]
        if de:
            lineas.append(
                f"  {condicion}: HMM = {_media([f['exactitud_desconocidas'] for f in de]):.3f}, "
                f"base raras = {_media([f['exactitud_desconocidas_base'] for f in de]):.3f}, "
                f"base firma = {_media([f['exactitud_base_firma'] for f in de]):.3f}, "
                f"desconocidas por corrida = {_media([f['n_desconocidas'] for f in de]):.1f}, "
                f"exactitud global = {_media([f['exactitud'] for f in de]):.3f}"
            )
            if condicion == "retenido":
                lineas.append(
                    f"    solo retenidas: HMM = {_media([f['exactitud_retenidas'] for f in de]):.3f} "
                    f"({_media([f['n_retenidas_eval'] for f in de]):.1f} apariciones por corrida)"
                )
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E4: curvas NB vs. RL, HMM en palabras desconocidas y confusión.")
    p.add_argument("--train", default=None,
                   help=f"corpus de entrenamiento (JSONL, C5); por omisión {TRAIN_POR_OMISION.relative_to(RAIZ)}")
    p.add_argument("--eval", default=None,
                   help=f"corpus de evaluación (JSONL, C5); por omisión {EVAL_POR_OMISION.relative_to(RAIZ)}")
    p.add_argument("--fixtures", action="store_true",
                   help="usa tests/fixtures/mini_corpus.jsonl como train y eval, con tokens_fixture")
    p.add_argument("--semillas", type=int, default=5, help="número de semillas (0..N-1)")
    p.add_argument("--salida", default=str(SALIDA_POR_OMISION), help="directorio de las CSV")
    args = p.parse_args(argv)
    if args.semillas < 1:
        p.error("--semillas debe ser ≥ 1")

    if args.fixtures:
        if args.train is not None or args.eval is not None:
            p.error("--fixtures no admite --train/--eval")
        rutas, modo = {"train": FIXTURE_CORPUS, "eval": FIXTURE_CORPUS}, "fixtures"
    else:
        rutas = {
            "train": args.train if args.train is not None else TRAIN_POR_OMISION,
            "eval": args.eval if args.eval is not None else EVAL_POR_OMISION,
        }
        modo = "tokenizar"

    try:
        filas = [f for s in range(args.semillas) for f in correr(s, rutas, modo)]
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    escritas = escribir_csv(filas, args.salida)
    print(resumen(filas))
    print("CSV: " + ", ".join(str(r) for r in escritas.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
