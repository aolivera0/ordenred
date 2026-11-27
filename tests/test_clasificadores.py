"""Pruebas de clasificadores.py (M3, contrato C6). Dueño: rol 4.

El ejemplo de `TestNaiveBayesEjemplo` es el ejercicio de la sustentación: 3 clases, 6 órdenes y
suavizado de Laplace. Las probabilidades esperadas se escriben a mano con `Fraction`, sin llamar al
código que se prueba.
"""
import json
import math
import re
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from tests import FIXTURES, RAIZ, tokens_fixture
from ordenred.clasificadores import NaiveBayes, RegresionLogistica, _normalizar_log
from ordenred.corpus import cargar_jsonl

CORPUS = cargar_jsonl(FIXTURES / "mini_corpus.jsonl")

# Ejemplo de la sustentación: órdenes ya convertidas a terminales.
X_EJEMPLO = [
    ["bloquea", "ssh"],
    ["bloquea", "ping"],
    ["deniega", "ssh"],
    ["permite", "http", "desde", "<CIDR>"],
    ["permite", "ssh"],
    ["limita", "http", "a", "<NUM>"],
]
Y_EJEMPLO = ["BLOQUEAR", "BLOQUEAR", "BLOQUEAR", "PERMITIR", "PERMITIR", "LIMITAR"]

# Conjunto separable para la regresión logística: el verbo decide la clase.
X_SEPARABLE = [
    ["bloquea", "ssh"],
    ["bloquea", "http", "desde", "<IP>"],
    ["bloquea", "el", "ping"],
    ["permite", "ssh", "desde", "<CIDR>"],
    ["permite", "http"],
    ["permite", "el", "trafico"],
    ["limita", "ssh", "a", "<NUM>"],
    ["limita", "http"],
    ["limita", "el", "ping", "a", "<NUM>"],
]
Y_SEPARABLE = ["BLOQUEAR"] * 3 + ["PERMITIR"] * 3 + ["LIMITAR"] * 3


def corpus_terminales():
    X = [[t.terminal for t in tokens_fixture(r["tokens"])] for r in CORPUS]
    y = [r["intencion"] for r in CORPUS]
    return X, y


class Temporal(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ruta = Path(self.dir.name) / "modelo.json"

    def tearDown(self):
        self.dir.cleanup()


class TestNormalizarLog(unittest.TestCase):
    def test_suma_uno_y_proporciones(self):
        p = _normalizar_log({"a": math.log(1.0), "b": math.log(3.0)})
        self.assertAlmostEqual(p["a"], 0.25, places=15)
        self.assertAlmostEqual(p["b"], 0.75, places=15)

    def test_sin_underflow(self):
        p = _normalizar_log({"a": -1000.0, "b": -1000.0 + math.log(2.0)})
        self.assertAlmostEqual(p["a"], 1 / 3, places=12)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=15)


class TestNaiveBayesEjemplo(unittest.TestCase):
    """Ejercicio de sustentación: clasificar «bloquea http desde <IP>» con la tabla de conteos.

    Vocabulario (|V| = 11): <CIDR> <NUM> a bloquea deniega desde http limita permite ping ssh.
    N = 6; N_B = 3, N_P = 2, N_L = 1; n_B = 6, n_P = 6, n_L = 4 palabras.
    «<IP>» no está en V: se ignora.
    """

    def setUp(self):
        self.nb = NaiveBayes(alfa=1.0)
        self.nb.entrenar(X_EJEMPLO, Y_EJEMPLO)

    def test_tabla_de_conteos(self):
        t = self.nb.tabla_conteos()
        self.assertEqual(t["N"], 6)
        self.assertEqual(t["N_c"], {"BLOQUEAR": 3, "LIMITAR": 1, "PERMITIR": 2})
        self.assertEqual(t["n_c"], {"BLOQUEAR": 6, "LIMITAR": 4, "PERMITIR": 6})
        self.assertEqual(t["V"], 11)
        self.assertEqual(t["n_cw"]["BLOQUEAR"]["bloquea"], 2)
        self.assertEqual(t["n_cw"]["BLOQUEAR"]["ssh"], 2)
        self.assertEqual(t["n_cw"]["PERMITIR"]["http"], 1)
        self.assertEqual(t["n_cw"]["LIMITAR"]["ssh"], 0)
        self.assertEqual(self.nb.clases, ["BLOQUEAR", "LIMITAR", "PERMITIR"])

    def test_probabilidades_calculadas_a_mano(self):
        # P(c) · Π P(w|c) con P(w|c) = (n_{c,w} + 1) / (n_c + 11); palabras: bloquea, http, desde.
        bloquear = F(3, 6) * F(2 + 1, 6 + 11) * F(0 + 1, 6 + 11) * F(0 + 1, 6 + 11)
        permitir = F(2, 6) * F(0 + 1, 6 + 11) * F(1 + 1, 6 + 11) * F(1 + 1, 6 + 11)
        limitar = F(1, 6) * F(0 + 1, 4 + 11) * F(1 + 1, 4 + 11) * F(0 + 1, 4 + 11)
        total = bloquear + permitir + limitar
        esperado = {"BLOQUEAR": bloquear / total, "PERMITIR": permitir / total, "LIMITAR": limitar / total}

        clase, probs = self.nb.predecir(["bloquea", "http", "desde", "<IP>"])
        self.assertEqual(clase, "BLOQUEAR")
        self.assertEqual(set(probs), set(esperado))
        for c, p in esperado.items():
            self.assertAlmostEqual(probs[c], float(p), places=12)

    def test_log_probabilidades_por_palabra(self):
        self.assertAlmostEqual(self.nb.log_prior["PERMITIR"], math.log(2 / 6), places=12)
        self.assertAlmostEqual(self.nb.log_verosimilitud["LIMITAR"]["a"], math.log(2 / 15), places=12)
        self.assertAlmostEqual(self.nb.log_verosimilitud["BLOQUEAR"]["http"], math.log(1 / 17), places=12)

    def test_repeticiones_cuentan(self):
        # Multinomial: «ssh ssh» multiplica dos veces P(ssh | c).
        bloquear = F(3, 6) * F(2 + 1, 17) ** 2
        permitir = F(2, 6) * F(1 + 1, 17) ** 2
        limitar = F(1, 6) * F(0 + 1, 15) ** 2
        total = bloquear + permitir + limitar
        _, probs = self.nb.predecir(["ssh", "ssh"])
        self.assertAlmostEqual(probs["BLOQUEAR"], float(bloquear / total), places=12)
        self.assertAlmostEqual(probs["LIMITAR"], float(limitar / total), places=12)


class TestNaiveBayes(Temporal):
    def test_probabilidades_suman_uno(self):
        nb = NaiveBayes()
        nb.entrenar(X_EJEMPLO, Y_EJEMPLO)
        for tokens in X_EJEMPLO + [[], ["zzz"], ["permite", "ssh", "ssh", "a"]]:
            _, probs = nb.predecir(tokens)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=12)
            self.assertEqual(set(probs), {"BLOQUEAR", "LIMITAR", "PERMITIR"})

    def test_fuera_de_vocabulario_da_el_prior(self):
        nb = NaiveBayes()
        nb.entrenar(X_EJEMPLO, Y_EJEMPLO)
        prior = {"BLOQUEAR": 3 / 6, "PERMITIR": 2 / 6, "LIMITAR": 1 / 6}
        for tokens in ([], ["zzz"], ["<IP>", "nada", "zzz"]):
            clase, probs = nb.predecir(tokens)
            self.assertEqual(clase, "BLOQUEAR")
            for c, p in prior.items():
                self.assertAlmostEqual(probs[c], p, places=12)
        self.assertEqual(nb.predecir(["ssh", "zzz"]), nb.predecir(["ssh"]))

    def test_una_sola_clase(self):
        nb = NaiveBayes()
        nb.entrenar([["bloquea", "ssh"], ["bloquea"]], ["BLOQUEAR", "BLOQUEAR"])
        self.assertEqual(nb.predecir(["permite", "http"]), ("BLOQUEAR", {"BLOQUEAR": 1.0}))
        self.assertEqual(nb.predecir([]), ("BLOQUEAR", {"BLOQUEAR": 1.0}))

    def test_errores(self):
        with self.assertRaises(ValueError):
            NaiveBayes().entrenar([], [])
        with self.assertRaises(ValueError):
            NaiveBayes().entrenar([["a"], ["b"]], ["X"])
        with self.assertRaises(RuntimeError):
            NaiveBayes().predecir(["a"])
        with self.assertRaises(RuntimeError):
            NaiveBayes().guardar(self.ruta)

    def test_empate_determinista(self):
        nb = NaiveBayes()
        nb.entrenar([["a"], ["b"]], ["Z", "A"])
        for _ in range(3):
            clase, probs = nb.predecir([])
            self.assertEqual(clase, "A")
            self.assertEqual(probs, {"A": 0.5, "Z": 0.5})
        self.assertEqual(nb.predecir(["c"])[0], "A")

    def test_empate_exacto_con_redondeo(self):
        # V = {u, v, w} (|V| = 3), alfa = 1. A: N_A = 4, n_A = 3, n_{A,w} = 2; Z: N_Z = 5, n_Z = 2, n_{Z,w} = 1.
        # P(A)·P(w|A) = 4/9 · 3/6 = 2/9 = 5/9 · 2/5 = P(Z)·P(w|Z): empate exacto, aunque los logaritmos
        # difieran en el último bit. Debe ganar "A", la primera en orden alfabético.
        self.assertEqual(F(4, 9) * F(2 + 1, 3 + 3), F(5, 9) * F(1 + 1, 2 + 3))
        X = [[], [], [], ["w", "w", "v"], [], [], [], [], ["w", "u"]]
        y = ["A"] * 4 + ["Z"] * 5
        nb = NaiveBayes()
        nb.entrenar(X, y)
        clase, probs = nb.predecir(["w"])
        self.assertEqual(clase, "A")
        self.assertAlmostEqual(probs["A"], 0.5, places=12)

    def test_ida_y_vuelta(self):
        nb = NaiveBayes(alfa=0.5)
        nb.entrenar(X_EJEMPLO, Y_EJEMPLO)
        nb.guardar(self.ruta)
        datos = json.loads(self.ruta.read_text(encoding="utf-8"))
        self.assertEqual(datos["tipo"], "NaiveBayes")
        self.assertEqual(datos["version"], 1)
        texto = self.ruta.read_text(encoding="utf-8")
        self.assertNotIn("Infinity", texto)
        self.assertNotIn("NaN", texto)
        for fila in datos["conteo_palabras"].values():
            for k in fila.values():
                self.assertIs(type(k), int)
        for k in datos["docs_por_clase"].values():
            self.assertIs(type(k), int)
        otro = NaiveBayes.cargar(self.ruta)
        self.assertEqual(otro.alfa, 0.5)
        self.assertEqual(otro.tabla_conteos(), nb.tabla_conteos())
        for tokens in X_EJEMPLO + [[], ["zzz"], ["bloquea", "http", "desde", "<IP>"]]:
            self.assertEqual(otro.predecir(tokens), nb.predecir(tokens))

    def test_cargar_otro_tipo_falla(self):
        rl = RegresionLogistica(epocas=5)
        rl.entrenar(X_EJEMPLO, Y_EJEMPLO)
        rl.guardar(self.ruta)
        with self.assertRaises(ValueError):
            NaiveBayes.cargar(self.ruta)

    def test_alfa_cero_sin_suavizado(self):
        nb = NaiveBayes(alfa=0.0)
        nb.entrenar(X_EJEMPLO, Y_EJEMPLO)
        clase, probs = nb.predecir(["limita"])
        self.assertEqual(clase, "LIMITAR")
        self.assertEqual(probs["BLOQUEAR"], 0.0)
        self.assertAlmostEqual(probs["LIMITAR"], 1.0, places=15)


class TestRegresionLogistica(Temporal):
    def test_gradiente_vs_diferencias_centrales(self):
        rng = np.random.default_rng(7)
        n, d, k = 6, 4, 3
        X = np.zeros((n, d + 1))
        X[:, :d] = rng.integers(0, 3, size=(n, d))
        X[:, d] = 1.0
        Y = np.zeros((n, k))
        for i, j in enumerate(rng.integers(0, k, size=n)):
            Y[i, j] = 1.0
        W = rng.normal(0.0, 0.5, size=(d + 1, k))
        rl = RegresionLogistica(l2=0.3)
        _, analitico = rl._perdida_y_gradiente(W, X, Y)
        numerico = np.zeros_like(W)
        h = 1e-5
        for a in range(W.shape[0]):
            for b in range(W.shape[1]):
                mas, menos = W.copy(), W.copy()
                mas[a, b] += h
                menos[a, b] -= h
                numerico[a, b] = (rl._perdida_y_gradiente(mas, X, Y)[0] - rl._perdida_y_gradiente(menos, X, Y)[0]) / (2 * h)
        dif = analitico - numerico
        error = math.sqrt((dif * dif).sum()) / (math.sqrt((analitico * analitico).sum()) + math.sqrt((numerico * numerico).sum()))
        self.assertLess(error, 1e-6)

    def test_perdida_calculada_a_mano(self):
        # Dos palabras y el sesgo en la última columna de X (última fila de W).
        X = np.array([[1.0, 2.0, 1.0], [0.0, 1.0, 1.0]])
        Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rl = RegresionLogistica(l2=0.5)
        W = np.zeros((3, 3))
        self.assertAlmostEqual(rl._perdida_y_gradiente(W, X, Y)[0], math.log(3), places=12)
        # W[0, 0] = 1 es el peso de la palabra 0 en la clase 0 (no el sesgo): z_1 = (1, 0, 0), z_2 = (0, 0, 0).
        W[0, 0] = 1.0
        esperado = (-(1 - math.log(math.e + 2)) - (0 - math.log(3))) / 2 + 0.5 / 2 * 1.0
        self.assertAlmostEqual(rl._perdida_y_gradiente(W, X, Y)[0], esperado, places=12)

    def test_sesgo_sin_l2(self):
        # Palabra ausente (columna 0) y sesgo igual en ambas clases: P es uniforme y el término de datos
        # del gradiente se anula. Si el sesgo se regularizara, su gradiente sería 5 · 3 = 15, no 0.
        X = np.array([[0.0, 1.0], [0.0, 1.0]])
        Y = np.array([[1.0, 0.0], [0.0, 1.0]])
        W = np.array([[2.0, -1.0], [3.0, 3.0]])
        _, g = RegresionLogistica(l2=5.0)._perdida_y_gradiente(W, X, Y)
        self.assertTrue(np.allclose(g[1], 0.0))
        self.assertTrue(np.allclose(g[0], 5.0 * W[0]))

    def test_perdida_no_creciente(self):
        rl = RegresionLogistica(tasa=0.05, epocas=150)
        rl.entrenar(X_SEPARABLE, Y_SEPARABLE)
        h = rl.historial_perdida
        self.assertEqual(len(h), 150)
        for antes, despues in zip(h, h[1:]):
            self.assertLessEqual(despues, antes + 1e-12)
        self.assertLess(h[-1], h[0])

    def test_separable_cien_por_ciento(self):
        rl = RegresionLogistica()
        rl.entrenar(X_SEPARABLE, Y_SEPARABLE)
        for tokens, c in zip(X_SEPARABLE, Y_SEPARABLE):
            clase, probs = rl.predecir(tokens)
            self.assertEqual(clase, c)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=12)
        self.assertEqual(rl.predecir(["limita", "ftp"])[0], "LIMITAR")

    def test_semilla(self):
        a, b, c = RegresionLogistica(semilla=1), RegresionLogistica(semilla=1), RegresionLogistica(semilla=2)
        for m in (a, b, c):
            m.entrenar(X_SEPARABLE, Y_SEPARABLE)
        self.assertTrue(np.array_equal(a.W, b.W))
        self.assertFalse(np.array_equal(a.W, c.W))
        # Sin épocas, W es exactamente la inicialización con la semilla: (|V| + 1) × K = 3 × 2.
        sin_entrenar = RegresionLogistica(epocas=0, semilla=9)
        sin_entrenar.entrenar([["a"], ["b"]], ["X", "Y"])
        self.assertTrue(np.array_equal(sin_entrenar.W, np.random.default_rng(9).normal(0.0, 0.01, size=(3, 2))))
        self.assertEqual(sin_entrenar.historial_perdida, [])

    def test_l2_reduce_la_norma(self):
        def norma(m):
            return float((m.W[:-1] * m.W[:-1]).sum())

        debil, fuerte = RegresionLogistica(l2=0.0, epocas=300), RegresionLogistica(l2=0.5, epocas=300)
        debil.entrenar(X_SEPARABLE, Y_SEPARABLE)
        fuerte.entrenar(X_SEPARABLE, Y_SEPARABLE)
        self.assertLess(norma(fuerte), norma(debil))

    def test_fuera_de_vocabulario_aporta_cero(self):
        rl = RegresionLogistica()
        rl.entrenar(X_SEPARABLE, Y_SEPARABLE)
        self.assertEqual(rl.predecir(["zzz"]), rl.predecir([]))
        self.assertEqual(rl.predecir(["bloquea", "zzz", "<RANGO>"]), rl.predecir(["bloquea"]))
        fila = rl.vectorizar([["zzz", "ssh", "ssh"]])[0]
        self.assertEqual(fila.sum(), 3.0)  # dos ssh + sesgo
        self.assertEqual(fila[-1], 1.0)
        self.assertEqual(fila[rl.indice["ssh"]], 2.0)

    def test_una_sola_clase(self):
        rl = RegresionLogistica()
        rl.entrenar([["bloquea"], ["bloquea", "ssh"]], ["BLOQUEAR", "BLOQUEAR"])
        self.assertIsNone(rl.W)
        self.assertEqual(rl.predecir(["permite"]), ("BLOQUEAR", {"BLOQUEAR": 1.0}))
        rl.guardar(self.ruta)
        self.assertEqual(RegresionLogistica.cargar(self.ruta).predecir([]), ("BLOQUEAR", {"BLOQUEAR": 1.0}))

    def test_errores(self):
        with self.assertRaises(ValueError):
            RegresionLogistica().entrenar([], [])
        with self.assertRaises(ValueError):
            RegresionLogistica().entrenar([["a"]], ["X", "Y"])
        with self.assertRaises(RuntimeError):
            RegresionLogistica().predecir(["a"])
        with self.assertRaises(RuntimeError):
            RegresionLogistica().guardar(self.ruta)

    def test_ida_y_vuelta(self):
        rl = RegresionLogistica(tasa=0.2, epocas=50, l2=0.01, semilla=3)
        rl.entrenar(X_SEPARABLE, Y_SEPARABLE)
        rl.guardar(self.ruta)
        texto = self.ruta.read_text(encoding="utf-8")
        datos = json.loads(texto)
        self.assertEqual((datos["tipo"], datos["version"]), ("RegresionLogistica", 1))
        self.assertNotIn("Infinity", texto)
        self.assertNotIn("NaN", texto)
        otro = RegresionLogistica.cargar(self.ruta)
        self.assertTrue(np.array_equal(otro.W, rl.W))
        self.assertEqual((otro.tasa, otro.epocas, otro.l2, otro.semilla), (0.2, 50, 0.01, 3))
        self.assertEqual(otro.historial_perdida, rl.historial_perdida)
        for tokens in X_SEPARABLE + [[], ["zzz"], ["permite", "ping", "a", "<NUM>"]]:
            self.assertEqual(otro.predecir(tokens), rl.predecir(tokens))

    def test_cargar_otro_tipo_falla(self):
        nb = NaiveBayes()
        nb.entrenar(X_EJEMPLO, Y_EJEMPLO)
        nb.guardar(self.ruta)
        with self.assertRaises(ValueError):
            RegresionLogistica.cargar(self.ruta)

    def test_cargar_rechaza_w_nulo_con_varias_clases(self):
        rl = RegresionLogistica(epocas=5)
        rl.entrenar(X_SEPARABLE, Y_SEPARABLE)
        rl.guardar(self.ruta)
        datos = json.loads(self.ruta.read_text(encoding="utf-8"))
        self.assertGreater(len(datos["clases"]), 1)
        datos["W"] = None
        self.ruta.write_text(json.dumps(datos), encoding="utf-8")
        with self.assertRaises(ValueError):
            RegresionLogistica.cargar(self.ruta)

    def test_cargar_json_mal_formado_da_value_error(self):
        for contenido in ([1, 2], {"tipo": "RegresionLogistica", "version": 1}):
            self.ruta.write_text(json.dumps(contenido), encoding="utf-8")
            with self.subTest(contenido=contenido), self.assertRaises(ValueError):
                RegresionLogistica.cargar(self.ruta)


class TestMiniCorpus(unittest.TestCase):
    def test_ambos_aciertan_todo_el_entrenamiento(self):
        X, y = corpus_terminales()
        self.assertIn("<CIDR>", X[0])
        for modelo in (NaiveBayes(), RegresionLogistica()):
            modelo.entrenar(X, y)
            aciertos = sum(modelo.predecir(tokens)[0] == c for tokens, c in zip(X, y))
            self.assertEqual(aciertos, len(y), type(modelo).__name__)


class TestReglasDelProyecto(unittest.TestCase):
    def test_sin_funciones_prohibidas_de_numpy(self):
        fuente = (RAIZ / "src" / "ordenred" / "clasificadores.py").read_text(encoding="utf-8")
        for prohibido in ("linalg", "unique", "bincount", "logaddexp", "scipy", "sklearn", "NotImplementedError"):
            self.assertIsNone(re.search(prohibido, fuente), prohibido)


if __name__ == "__main__":
    unittest.main()
