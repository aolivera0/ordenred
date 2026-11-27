"""Pruebas de hmm.py (M4, contrato C6). Dueño: rol 4.

Las probabilidades esperadas se calculan aquí a mano con `fractions.Fraction` a partir de los conteos,
sin llamar al código del HMM; Viterbi se compara con la enumeración exhaustiva de todas las secuencias.
"""
import itertools
import json
import math
import random
import tempfile
import unittest
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

from tests import FIXTURES, tokens_fixture
from ordenred.corpus import cargar_jsonl
from ordenred.cyk import ParserCYK
from ordenred.hmm import HMM, firma
from ordenred.pcfg import PCFG

# Corpus de juguete: 3 oraciones, etiquetas D (determinante), N (nombre), V (verbo).
CORPUS_3 = [
    [("bloquea", "V"), ("ssh", "N")],
    [("permite", "V"), ("el", "D"), ("ssh", "N")],
    [("bloquea", "V"), ("el", "D"), ("http", "N")],
]


def entrenado(oraciones, **kw) -> HMM:
    h = HMM(**kw)
    h.entrenar(oraciones)
    return h


def log(fr: F) -> float:
    return math.log(fr.numerator) - math.log(fr.denominator)


class TestFirma(unittest.TestCase):
    def test_firmas(self):
        casos = {
            "eth0": "<UNK-DIG>",
            "10.0.0.5": "<UNK-DIG>",
            "conexiones": "<UNK-es>",  # termina en «-iones», no en «-ciones»
            "conexion": "<UNK>",
            "direcciones": "<UNK-ciones>",
            "direccion": "<UNK-cion>",
            "rapidamente": "<UNK-mente>",
            "bloqueando": "<UNK-ando>",
            "permitiendo": "<UNK-iendo>",
            "denegar": "<UNK-ar>",
            "mover": "<UNK-er>",
            "impedir": "<UNK-ir>",
            "redes": "<UNK-es>",
            "puertos": "<UNK-os>",
            "reglas": "<UNK-as>",
            "deniega": "<UNK-a>",
            "puerto": "<UNK-o>",
            "web": "<UNK>",
            "http": "<UNK>",
        }
        for palabra, esperada in casos.items():
            with self.subTest(palabra=palabra):
                self.assertEqual(firma(palabra), esperada)


class TestConteosYProbabilidades(unittest.TestCase):
    """CORPUS_3 con umbral_raras = 1: «permite» y «http» son hapax y se entrenan como su firma."""

    def setUp(self):
        self.h = entrenado(CORPUS_3)  # alfa = 1, usar_fin = True

    def test_conteos(self):
        h = self.h
        self.assertEqual(h.etiquetas, ["D", "N", "V"])
        self.assertEqual(h.n_oraciones, 3)
        self.assertEqual(h.conteo_inicio, {"D": 0, "N": 0, "V": 3})
        self.assertEqual(h.conteo_trans["V"], {"D": 2, "N": 1, "V": 0})
        self.assertEqual(h.conteo_trans["D"], {"D": 0, "N": 2, "V": 0})
        self.assertEqual(h.conteo_trans["N"], {"D": 0, "N": 0, "V": 0})
        self.assertEqual(h.conteo_fin, {"D": 0, "N": 3, "V": 0})
        self.assertEqual(h.conteo_etiqueta, {"D": 2, "N": 3, "V": 3})
        # Reemplazo de hapax: «permite» -> <UNK-e>, «http» -> <UNK>; no se cuentan como ellas mismas.
        self.assertEqual(h.conteo_emision["V"], {"bloquea": 2, "<UNK-e>": 1})
        self.assertEqual(h.conteo_emision["D"], {"el": 2})
        self.assertEqual(h.conteo_emision["N"], {"ssh": 2, "<UNK>": 1})
        self.assertEqual(h.conocidas, {"bloquea", "ssh", "el"})
        self.assertEqual(h.vocabulario, sorted(["bloquea", "el", "ssh", "<UNK-e>", "<UNK>"]))
        self.assertTrue(h.es_conocida("bloquea"))
        self.assertFalse(h.es_conocida("permite"))
        self.assertFalse(h.es_conocida("<UNK-e>"))

    def test_probabilidades_con_fin(self):
        h = self.h
        T, N, V_ = 3, 3, 5  # etiquetas, oraciones, |V'|
        pi = {"D": F(0 + 1, N + T), "N": F(0 + 1, N + T), "V": F(3 + 1, N + T)}  # 1/6, 1/6, 4/6
        # Con FIN el denominador es C(i) + α(T+1) = C(i) + 4.
        A = {
            "V": {"D": F(2 + 1, 3 + 4), "N": F(1 + 1, 3 + 4), "V": F(0 + 1, 3 + 4)},
            "D": {"D": F(1, 2 + 4), "N": F(2 + 1, 2 + 4), "V": F(1, 2 + 4)},
            "N": {"D": F(1, 3 + 4), "N": F(1, 3 + 4), "V": F(1, 3 + 4)},
        }
        fin = {"V": F(0 + 1, 3 + 4), "D": F(0 + 1, 2 + 4), "N": F(3 + 1, 3 + 4)}
        B = {
            ("V", "bloquea"): F(2 + 1, 3 + V_),
            ("V", "<UNK-e>"): F(1 + 1, 3 + V_),
            ("V", "ssh"): F(0 + 1, 3 + V_),
            ("D", "el"): F(2 + 1, 2 + V_),
            ("D", "<UNK>"): F(0 + 1, 2 + V_),
            ("N", "ssh"): F(2 + 1, 3 + V_),
            ("N", "<UNK>"): F(1 + 1, 3 + V_),
        }
        for t in "DNV":
            self.assertAlmostEqual(sum(A[t].values()) + fin[t], 1)  # verificación de la propia prueba
        e, o = h.indice_etiqueta, h.indice_obs
        for t, p in pi.items():
            self.assertAlmostEqual(h.log_pi[e[t]], log(p), places=12)
        for i in A:
            self.assertAlmostEqual(h.log_fin[e[i]], log(fin[i]), places=12)
            for j, p in A[i].items():
                self.assertAlmostEqual(h.log_A[e[i], e[j]], log(p), places=12)
        for (t, w), p in B.items():
            self.assertAlmostEqual(h.log_B[e[t], o[w]], log(p), places=12)

    def test_probabilidades_sin_fin(self):
        h = entrenado(CORPUS_3, usar_fin=False, alfa_trans=0.5)
        a = F(1, 2)
        # Sin FIN: (C(i,j) + α) / (C(i→·) + α·T), con C(V→·) = 3, C(D→·) = 2, C(N→·) = 0.
        A = {
            ("V", "D"): (2 + a) / (3 + 3 * a),
            ("V", "V"): (0 + a) / (3 + 3 * a),
            ("D", "N"): (2 + a) / (2 + 3 * a),
            ("N", "N"): (0 + a) / (0 + 3 * a),
        }
        e = h.indice_etiqueta
        for (i, j), p in A.items():
            self.assertAlmostEqual(h.log_A[e[i], e[j]], log(p), places=12)
        self.assertAlmostEqual(h.log_pi[e["V"]], log((3 + a) / (3 + 3 * a)), places=12)

    def test_logprob_conjunta_a_mano(self):
        # «bloquea el ssh» con V D N y FIN: π_V · B_V(bloquea) · A_VD · B_D(el) · A_DN · B_N(ssh) · fin_N
        p = F(4, 6) * F(3, 8) * F(3, 7) * F(3, 7) * F(3, 6) * F(3, 8) * F(4, 7)
        self.assertAlmostEqual(self.h.logprob_conjunta(["bloquea", "el", "ssh"], ["V", "D", "N"]), log(p), places=12)
        # Palabra nueva «deniega» -> <UNK> (no se vio <UNK-a>); «permite» -> <UNK-e>.
        p2 = F(4, 6) * F(1, 8) * F(2, 7) * F(1, 8) * F(4, 7)
        self.assertAlmostEqual(self.h.logprob_conjunta(["deniega", "permite"], ["V", "N"]), log(p2), places=12)
        self.assertEqual(self.h.logprob_conjunta([], []), 0.0)


class TestTablaViterbiSustentacion(unittest.TestCase):
    """Tabla de Viterbi de 3 palabras × 3 etiquetas, sin FIN, umbral_raras = 0 (sin firmas).

    Corpus: «bloquea el ssh» (V D N), «bloquea ssh» (V N), «permite el http» (V D N).
    V' = {bloquea, el, ssh, permite, http, <UNK>} (6); C(D) = 2, C(N) = 3, C(V) = 3.
    π:  D = 1/6, N = 1/6, V = 4/6                                     [(C(INI,j)+1)/(3+3)]
    A:  V→(D,N,V) = 3/6, 2/6, 1/6;  D→(D,N,V) = 1/5, 3/5, 1/5;  N→· = 1/3 cada una   [(C(i,j)+1)/(C(i→·)+3)]
    B:  B_t(w) = (C(t,w)+1)/(C(t)+6)
    """

    CORPUS = [
        [("bloquea", "V"), ("el", "D"), ("ssh", "N")],
        [("bloquea", "V"), ("ssh", "N")],
        [("permite", "V"), ("el", "D"), ("http", "N")],
    ]

    def test_tabla_delta(self):
        h = entrenado(self.CORPUS, usar_fin=False, umbral_raras=0)
        self.assertEqual(h.etiquetas, ["D", "N", "V"])
        # Fila 1 «bloquea»: δ1(j) = π_j · B_j(bloquea)
        d1 = {"D": F(1, 6) * F(1, 8), "N": F(1, 6) * F(1, 9), "V": F(4, 6) * F(3, 9)}
        # Fila 2 «el»: δ2(j) = max_i δ1(i) · A_ij · B_j(el); en las tres columnas gana i = V.
        d2 = {
            "D": max(d1["D"] * F(1, 5), d1["N"] * F(1, 3), d1["V"] * F(3, 6)) * F(3, 8),
            "N": max(d1["D"] * F(3, 5), d1["N"] * F(1, 3), d1["V"] * F(2, 6)) * F(1, 9),
            "V": max(d1["D"] * F(1, 5), d1["N"] * F(1, 3), d1["V"] * F(1, 6)) * F(1, 9),
        }
        # Fila 3 «ssh»: δ3(j) = max_i δ2(i) · A_ij · B_j(ssh)
        d3 = {
            "D": max(d2["D"] * F(1, 5), d2["N"] * F(1, 3), d2["V"] * F(3, 6)) * F(1, 8),
            "N": max(d2["D"] * F(3, 5), d2["N"] * F(1, 3), d2["V"] * F(2, 6)) * F(3, 9),
            "V": max(d2["D"] * F(1, 5), d2["N"] * F(1, 3), d2["V"] * F(1, 6)) * F(1, 9),
        }
        # Valores cerrados (para llenar la tabla en la sustentación).
        self.assertEqual(d1, {"D": F(1, 48), "N": F(1, 54), "V": F(2, 9)})
        self.assertEqual(d2, {"D": F(1, 24), "N": F(2, 243), "V": F(1, 243)})
        self.assertEqual(d3, {"D": F(1, 960), "N": F(1, 120), "V": F(1, 1080)})

        palabras = ["bloquea", "el", "ssh"]
        delta = h.tabla_viterbi(palabras)
        self.assertEqual(delta.shape, (3, 3))
        for k, fila in enumerate([d1, d2, d3]):
            for t, valor in fila.items():
                with self.subTest(fila=k, etiqueta=t):
                    self.assertAlmostEqual(delta[k, h.indice_etiqueta[t]], log(valor), places=12)
        etiquetas, lp = h.viterbi(palabras)
        self.assertEqual(etiquetas, ["V", "D", "N"])
        self.assertAlmostEqual(lp, log(F(1, 120)), places=12)
        # Punteros de retroceso: N en la fila 3 viene de D; D en la fila 2 viene de V.
        self.assertEqual(h.psi[2, h.indice_etiqueta["N"]], h.indice_etiqueta["D"])
        self.assertEqual(h.psi[1, h.indice_etiqueta["D"]], h.indice_etiqueta["V"])


def corpus_aleatorio(rnd: random.Random, etiquetas: list[str]) -> list[list[tuple[str, str]]]:
    palabras = ["a", "b", "c", "d", "redes", "puertos", "eth0", "web", "<IP>"]
    return [
        [(rnd.choice(palabras), rnd.choice(etiquetas)) for _ in range(rnd.randint(1, 5))]
        for _ in range(rnd.randint(2, 7))
    ]


def frase_aleatoria(rnd: random.Random) -> list[str]:
    palabras = ["a", "b", "c", "d", "redes", "puertos", "eth0", "web", "<IP>", "zeta", "vlan9", "nuevas", "x"]
    return [rnd.choice(palabras) for _ in range(rnd.randint(1, 4))]


def logprob_independiente(corpus, palabras, etiquetas, alfa_trans, alfa_emis, umbral_raras, usar_fin) -> float:
    """log P(palabras, etiquetas) recontando desde el corpus con Fraction, sin usar el HMM (solo `firma`)."""
    a_t, a_e = F(alfa_trans), F(alfa_emis)  # conversión exacta del float
    especiales = {"<IP>", "<CIDR>", "<NUM>", "<RANGO>", "<IFAZ>"}
    frec = Counter(w for o in corpus for w, _ in o)
    rara = {w for w in frec if frec[w] <= umbral_raras and w not in especiales}
    conocidas = set(frec) - rara
    V = conocidas | {firma(w) for w in rara} | {"<UNK>"}
    tags = sorted({e for o in corpus for _, e in o})
    T, N = len(tags), len(corpus)
    ini, trans, fin, ct, em = Counter(), Counter(), Counter(), Counter(), Counter()
    for o in corpus:
        ini[o[0][1]] += 1
        fin[o[-1][1]] += 1
        for k, (w, e) in enumerate(o):
            ct[e] += 1
            em[(e, firma(w) if w in rara else w)] += 1
            if k:
                trans[(o[k - 1][1], e)] += 1

    def obs(w):
        if w in conocidas:
            return w
        return firma(w) if firma(w) in V else "<UNK>"

    def A(i, j):
        if usar_fin:
            return (trans[(i, j)] + a_t) / (ct[i] + a_t * (T + 1))
        return (trans[(i, j)] + a_t) / (sum(trans[(i, x)] for x in tags) + a_t * T)

    def B(t, w):
        return (em[(t, obs(w))] + a_e) / (ct[t] + a_e * len(V))

    p = (ini[etiquetas[0]] + a_t) / (N + a_t * T) * B(etiquetas[0], palabras[0])
    for k in range(1, len(palabras)):
        p *= A(etiquetas[k - 1], etiquetas[k]) * B(etiquetas[k], palabras[k])
    if usar_fin:
        p *= (fin[etiquetas[-1]] + a_t) / (ct[etiquetas[-1]] + a_t * (T + 1))
    return log(p)


class TestViterbiContraEnumeracion(unittest.TestCase):
    def mejor_por_enumeracion(self, h, palabras, permitidas=None):
        mejor = -math.inf
        for secuencia in itertools.product(h.etiquetas, repeat=len(palabras)):
            if permitidas and any(secuencia[p] not in s for p, s in permitidas.items()):
                continue
            mejor = max(mejor, h.logprob_conjunta(palabras, list(secuencia)))
        return mejor

    def test_veinte_modelos_aleatorios_con_y_sin_fin(self):
        modelos = 0
        for semilla in range(20):
            rnd = random.Random(semilla)
            etiquetas = ["A", "B", "C"][: rnd.randint(1, 3)]
            corpus = corpus_aleatorio(rnd, etiquetas)
            kw = dict(alfa_trans=rnd.choice([0.1, 0.5, 1.0, 2.0]), alfa_emis=rnd.choice([0.5, 1.0]),
                      umbral_raras=rnd.randint(0, 2))
            for usar_fin in (True, False):
                h = entrenado(corpus, usar_fin=usar_fin, **kw)
                modelos += 1
                for _ in range(5):
                    palabras = frase_aleatoria(rnd)
                    with self.subTest(semilla=semilla, usar_fin=usar_fin, palabras=palabras):
                        etiq, lp = h.viterbi(palabras)
                        optimo = self.mejor_por_enumeracion(h, palabras)
                        self.assertEqual(len(etiq), len(palabras))
                        self.assertAlmostEqual(lp, optimo, places=9)
                        self.assertAlmostEqual(h.logprob_conjunta(palabras, etiq), optimo, places=9)
        self.assertGreaterEqual(modelos, 20)

    def test_enumeracion_con_logprob_independiente(self):
        """La enumeración no depende de `logprob_conjunta`: se recalcula todo desde el corpus con Fraction."""
        for semilla in range(8):
            rnd = random.Random(500 + semilla)
            etiquetas = ["A", "B", "C"][: rnd.randint(2, 3)]
            corpus = corpus_aleatorio(rnd, etiquetas)
            kw = dict(alfa_trans=rnd.choice([0.5, 1.0, 2.0]), alfa_emis=rnd.choice([0.25, 1.0]),
                      umbral_raras=rnd.randint(0, 2))
            for usar_fin in (True, False):
                h = entrenado(corpus, usar_fin=usar_fin, **kw)
                for _ in range(3):
                    palabras = frase_aleatoria(rnd)
                    with self.subTest(semilla=semilla, usar_fin=usar_fin, palabras=palabras):
                        independientes = {
                            sec: logprob_independiente(corpus, palabras, sec, usar_fin=usar_fin, **kw)
                            for sec in itertools.product(h.etiquetas, repeat=len(palabras))
                        }
                        for sec, lp in independientes.items():
                            self.assertAlmostEqual(h.logprob_conjunta(palabras, list(sec)), lp, places=9)
                        etiq, lp = h.viterbi(palabras)
                        self.assertAlmostEqual(lp, max(independientes.values()), places=9)
                        self.assertAlmostEqual(independientes[tuple(etiq)], lp, places=9)

    def test_restriccion_de_etiquetas_contra_enumeracion(self):
        for semilla in range(20):
            rnd = random.Random(100 + semilla)
            h = entrenado(corpus_aleatorio(rnd, ["A", "B", "C"]), usar_fin=rnd.random() < 0.5)
            palabras = frase_aleatoria(rnd)
            pos = rnd.randrange(len(palabras))
            permitidas = set(rnd.sample(h.etiquetas, rnd.randint(1, len(h.etiquetas))))
            with self.subTest(semilla=semilla, palabras=palabras, pos=pos, permitidas=permitidas):
                etiq, lp = h._decodificar(palabras, {pos: permitidas})
                self.assertIn(etiq[pos], permitidas)
                self.assertAlmostEqual(lp, self.mejor_por_enumeracion(h, palabras, {pos: permitidas}), places=9)
                self.assertEqual(h.etiquetas_unk(palabras, [pos], permitidas), {pos: etiq[pos]})


class TestPermitidasRegresion(unittest.TestCase):
    """Regresión: `etiquetas_unk` nunca devuelve una etiqueta fuera de `permitidas` (C6)."""

    def test_siempre_dentro_de_permitidas(self):
        for semilla in range(300):
            rnd = random.Random(1000 + semilla)
            h = entrenado(corpus_aleatorio(rnd, ["A", "B", "C", "D"]), usar_fin=rnd.random() < 0.5,
                          alfa_trans=rnd.choice([1e-6, 0.01, 1.0]), alfa_emis=rnd.choice([1e-6, 0.01, 1.0]))
            palabras = frase_aleatoria(rnd)
            posiciones = rnd.sample(range(len(palabras)), rnd.randint(1, len(palabras)))
            permitidas = set(rnd.sample(h.etiquetas, rnd.randint(1, len(h.etiquetas)))) | {"OTRA"}
            with self.subTest(semilla=semilla):
                propuesta = h.etiquetas_unk(palabras, posiciones, permitidas)
                self.assertEqual(set(propuesta), set(posiciones))
                self.assertTrue(set(propuesta.values()) <= permitidas, propuesta)

    def test_permitidas_excluye_la_etiqueta_mas_probable(self):
        # Con α > 0 hay un camino válido aunque la única etiqueta permitida sea la menos probable.
        h = entrenado([[("a", "A"), ("b", "B")], [("a", "D")]], alfa_trans=1e-9)
        self.assertEqual(h.etiquetas_unk(["a", "b"], [0, 1], {"D"}), {0: "D", 1: "D"})


class TestPalabrasDesconocidasYErrores(unittest.TestCase):
    CORPUS = [
        [("bloquea", "V"), ("el", "D"), ("puerto", "N"), ("<NUM>", "NUM")],
        [("permite", "V"), ("el", "D"), ("trafico", "N")],
        [("bloquea", "V"), ("la", "D"), ("interfaz", "N"), ("eth0", "IF")],
        [("bloquea", "V"), ("el", "D"), ("acceso", "N")],
        [("permite", "V"), ("la", "D"), ("vlan10", "IF")],
    ]

    def setUp(self):
        self.h = entrenado(self.CORPUS)

    def test_palabra_no_vista_recibe_su_firma_y_etiqueta_coherente(self):
        # puerto, trafico, acceso (hapax) -> <UNK-o> como N; «servicio» no se vio y cae en <UNK-o>.
        self.assertEqual(self.h.observacion("servicio"), "<UNK-o>")
        self.assertFalse(self.h.es_conocida("servicio"))
        self.assertEqual(self.h.viterbi(["bloquea", "el", "servicio"])[0], ["V", "D", "N"])

    def test_firma_con_digito(self):
        # eth0 y vlan10 (hapax) -> <UNK-DIG> como IF; «wlan1» nunca visto también.
        self.assertEqual(self.h.observacion("wlan1"), "<UNK-DIG>")
        self.assertEqual(self.h.viterbi(["permite", "la", "wlan1"])[0], ["V", "D", "IF"])

    def test_firma_no_vista_cae_en_unk(self):
        self.assertEqual(firma("ssh"), "<UNK>")
        self.assertEqual(self.h.observacion("ssh"), "<UNK>")
        self.assertEqual(self.h.observacion("deniegas"), "<UNK>")  # <UNK-as> no se vio

    def test_terminales_especiales_no_se_firman(self):
        # <NUM> aparece una sola vez y aun así queda como palabra conocida.
        self.assertTrue(self.h.es_conocida("<NUM>"))
        self.assertIn("<NUM>", self.h.conteo_emision["NUM"])
        self.assertEqual(self.h.observacion("<NUM>"), "<NUM>")

    def test_permitidas(self):
        palabras = ["bloquea", "el", "servicio"]
        self.assertEqual(self.h.etiquetas_unk(palabras, [2]), {2: "N"})
        self.assertIn(self.h.etiquetas_unk(palabras, [2], permitidas={"IF", "NUM"})[2], {"IF", "NUM"})
        self.assertEqual(self.h.etiquetas_unk(palabras, [2], permitidas={"N", "V"}), {2: "N"})
        self.assertEqual(self.h.etiquetas_unk(palabras, [2], permitidas={"NUM"}), {2: "NUM"})
        # Las etiquetas permitidas que el modelo no conoce se ignoran; si no queda ninguna, error.
        self.assertEqual(self.h.etiquetas_unk(palabras, [2], permitidas={"NUM", "XYZ"}), {2: "NUM"})
        with self.assertRaises(ValueError):
            self.h.etiquetas_unk(palabras, [2], permitidas={"XYZ"})
        # Varias posiciones y orden de la respuesta.
        self.assertEqual(self.h.etiquetas_unk(palabras, [2, 0]), {2: "N", 0: "V"})
        self.assertEqual(self.h.etiquetas_unk(palabras, []), {})

    def test_posiciones_fuera_de_rango(self):
        for pos in ([3], [-1], [0, 7]):
            with self.subTest(pos=pos), self.assertRaises(ValueError):
                self.h.etiquetas_unk(["bloquea", "el", "servicio"], pos)
        with self.assertRaises(ValueError):
            self.h.etiquetas_unk([], [0])

    def test_oracion_vacia(self):
        self.assertEqual(self.h.viterbi([]), ([], 0.0))
        self.assertEqual(self.h.etiquetas_unk([], []), {})

    def test_errores(self):
        with self.assertRaises(ValueError):
            HMM().entrenar([])
        with self.assertRaises(ValueError):
            HMM().entrenar([[], []])
        for llamada in (lambda h: h.viterbi(["a"]), lambda h: h.viterbi([]),
                        lambda h: h.etiquetas_unk(["a"], [0]), lambda h: h.logprob_conjunta(["a"], ["V"])):
            with self.assertRaises(RuntimeError):
                llamada(HMM())
        with self.assertRaises(ValueError):
            self.h.logprob_conjunta(["bloquea"], ["V", "D"])
        with self.assertRaises(ValueError):
            self.h.logprob_conjunta(["bloquea"], ["XYZ"])
        for kw in (dict(alfa_trans=0), dict(alfa_emis=0), dict(alfa_trans=0.0), dict(alfa_trans=-1),
                   dict(alfa_emis=-0.5), dict(alfa_trans=float("nan")), dict(alfa_emis=float("nan")),
                   dict(alfa_trans=float("inf"))):
            with self.subTest(**kw), self.assertRaises(ValueError):
                HMM(**kw)
        with self.assertRaises(ValueError):
            HMM(umbral_raras=-1)

    def test_reentrenar_reinicia_los_conteos(self):
        h = entrenado(self.CORPUS)
        h.entrenar(CORPUS_3)
        self.assertEqual(h.etiquetas, ["D", "N", "V"])
        self.assertEqual(h.n_oraciones, 3)


class TestPersistencia(unittest.TestCase):
    def test_ida_y_vuelta(self):
        for usar_fin in (True, False):
            rnd = random.Random(7)
            h = entrenado(corpus_aleatorio(rnd, ["A", "B", "C"]), usar_fin=usar_fin, alfa_trans=0.5,
                          alfa_emis=0.25, umbral_raras=2)
            with tempfile.TemporaryDirectory() as d:
                ruta = Path(d) / "sub" / "hmm.json"
                h.guardar(ruta)
                texto = ruta.read_text(encoding="utf-8")
                datos = json.loads(texto)
                h2 = HMM.cargar(ruta)
            self.assertEqual((datos["tipo"], datos["version"]), ("HMM", 1))
            self.assertNotIn("Infinity", texto)
            self.assertNotIn("NaN", texto)
            self.assertTrue(all(isinstance(c, int) for c in datos["conteo_inicio"].values()))
            self.assertEqual((h2.alfa_trans, h2.alfa_emis, h2.umbral_raras, h2.usar_fin), (0.5, 0.25, 2, usar_fin))
            self.assertEqual(h2.vocabulario, h.vocabulario)
            self.assertTrue((h2.log_A == h.log_A).all() and (h2.log_B == h.log_B).all())
            self.assertTrue((h2.log_pi == h.log_pi).all() and (h2.log_fin == h.log_fin).all())
            for _ in range(10):
                palabras = frase_aleatoria(rnd)
                self.assertEqual(h2.viterbi(palabras), h.viterbi(palabras))

    def test_cargar_valida_el_tipo(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "otro.json"
            ruta.write_text(json.dumps({"tipo": "NaiveBayes", "version": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                HMM.cargar(ruta)
            ruta.write_text(json.dumps({"tipo": "HMM", "version": 99}), encoding="utf-8")
            with self.assertRaises(ValueError):
                HMM.cargar(ruta)

    def test_cargar_json_mal_formado_da_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "hmm.json"
            entrenado(CORPUS_3).guardar(ruta)
            completo = json.loads(ruta.read_text(encoding="utf-8"))
            sin_campo = {k: v for k, v in completo.items() if k != "conteo_emision"}
            conteos_rotos = dict(completo, conteo_trans=[1, 2, 3])
            for contenido in ([1, 2, 3], "HMM", None, {"tipo": "HMM", "version": 1}, sin_campo, conteos_rotos):
                ruta.write_text(json.dumps(contenido), encoding="utf-8")
                with self.subTest(contenido=str(contenido)[:40]), self.assertRaises(ValueError):
                    HMM.cargar(ruta)

    def test_guardar_sin_entrenar(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(RuntimeError):
            HMM().guardar(Path(d) / "hmm.json")


class TestIntegracionConCYK(unittest.TestCase):
    """Integración: el HMM propone la etiqueta de una palabra fuera del léxico de la PCFG y el CYK la usa."""

    def test_palabra_fuera_del_lexico(self):
        corpus = cargar_jsonl(FIXTURES / "mini_corpus.jsonl")
        oraciones = [
            [(t.terminal, e) for t, e in zip(tokens_fixture(r["tokens"]), r["etiquetas"])] for r in corpus
        ]
        h = entrenado(oraciones)
        g = PCFG.desde_archivo(FIXTURES / "mini.pcfg")
        permitidas = {r.izq for r in g.reglas if r.es_lexica}

        terminales = [t.terminal for t in tokens_fixture(["deniega", "ssh"])]
        self.assertNotIn("deniega", g.indice_lexico)
        self.assertFalse(h.es_conocida("deniega"))
        # Se espera VACC: las 5 órdenes del corpus empiezan por VACC, así que π concentra la masa en VACC
        # (6/15 frente a 1/15 de cada otra etiqueta), y VACC → SERV ya se vio dos veces («bloquea ssh»,
        # «permite http»). La única rival por la firma <UNK-a> es PDEST («hacia», «a»), pero
        # π_PDEST = 1/15 y PDEST → SERV nunca se vio.
        propuesta = h.etiquetas_unk(terminales, [0], permitidas)
        self.assertEqual(propuesta, {0: "VACC"})

        arboles = ParserCYK(g).analizar(terminales, etiquetas_unk=propuesta)
        self.assertGreaterEqual(len(arboles), 1)
        self.assertEqual(arboles[0].etiqueta, "S")
        # Sin la propuesta del HMM, la orden no tiene análisis.
        self.assertEqual(ParserCYK(g).analizar(terminales), [])


if __name__ == "__main__":
    unittest.main()
