"""Pruebas de cyk.py (M5b). Dueño: rol 1.

La referencia es tests/fixtures/mini_cyk_esperado.json, calculado por enumeración exhaustiva
(tests/fixtures/generar_fixtures.py), independiente de este CYK.
"""
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path

from tests import FIXTURES, tokens_fixture
from ordenred.arbol import a_corchetes, hojas, reglas
from ordenred.corpus import cargar_jsonl
from ordenred.cyk import ParserCYK
from ordenred.pcfg import PCFG

GRAMATICA = PCFG.desde_archivo(FIXTURES / "mini.pcfg")
ESPERADO = json.loads((FIXTURES / "mini_cyk_esperado.json").read_text(encoding="utf-8"))["casos"]
CORPUS = cargar_jsonl(FIXTURES / "mini_corpus.jsonl")


def gramatica_desde_texto(texto: str) -> PCFG:
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "g.pcfg"
        ruta.write_text(texto, encoding="utf-8")
        return PCFG.desde_archivo(ruta)


def logprob_por_reglas(g: PCFG, arbol) -> float:
    """log P del árbol recalculada desde cero con sus reglas."""
    lp = {(r.izq, r.der): r.logprob for r in g.reglas}
    return math.fsum(lp[r] for r in reglas(arbol))


def con_limite_de_tiempo(funcion, segundos=5.0):
    resultado, error = [], []

    def correr():
        try:
            resultado.append(funcion())
        except Exception as e:  # pragma: no cover
            error.append(e)

    hilo = threading.Thread(target=correr, daemon=True)
    hilo.start()
    hilo.join(segundos)
    if hilo.is_alive():
        raise AssertionError(f"el análisis no terminó en {segundos} s (¿bucle infinito?)")
    if error:
        raise error[0]
    return resultado[0]


class TestCYKContraEsperado(unittest.TestCase):
    def setUp(self):
        self.parser = ParserCYK(GRAMATICA)

    def test_todos_los_arboles_y_sus_probabilidades(self):
        for caso in ESPERADO:
            with self.subTest(orden=caso["orden"]):
                arboles = self.parser.analizar(caso["terminales"], k=10)
                self.assertEqual(len(arboles), caso["n_arboles"])
                for a, p in zip(arboles, caso["probs_ordenadas"]):
                    self.assertAlmostEqual(math.exp(a.logprob) / p, 1.0, places=9)
                self.assertEqual(len({a_corchetes(a) for a in arboles}), len(arboles), "árboles repetidos")

    def test_logprob_coincide_con_las_reglas_del_arbol(self):
        for caso in ESPERADO:
            for a in self.parser.analizar(caso["terminales"], k=10):
                with self.subTest(orden=caso["orden"], arbol=a_corchetes(a)):
                    self.assertAlmostEqual(a.logprob, logprob_por_reglas(GRAMATICA, a), places=9)

    def test_el_mejor_arbol_es_el_de_oro_con_tokens_reales(self):
        for r in CORPUS:
            with self.subTest(orden=r["orden"]):
                tokens = tokens_fixture(r["tokens"])
                mejor = self.parser.analizar([t.terminal for t in tokens], k=1, tokens=tokens)[0]
                self.assertEqual(a_corchetes(mejor), r["arbol"])
                self.assertEqual(hojas(mejor), tokens)

    def test_los_k_mejores_para_cada_k(self):
        # Con empates, cuál de los árboles empatados entra en el corte depende de k;
        # lo que sí debe cumplirse: mismas probabilidades que el prefijo y árboles del conjunto total.
        terminales = ESPERADO[1]["terminales"]  # 5 árboles, con empates
        todos = self.parser.analizar(terminales, k=5)
        for k in range(1, 6):
            with self.subTest(k=k):
                obtenidos = self.parser.analizar(terminales, k=k)
                self.assertEqual(len(obtenidos), k)
                for a, b in zip(obtenidos, todos):
                    self.assertAlmostEqual(a.logprob, b.logprob, places=9)
                self.assertLessEqual({a_corchetes(a) for a in obtenidos}, {a_corchetes(a) for a in todos})


class TestCasosBorde(unittest.TestCase):
    def test_sin_analisis_y_vacio(self):
        p = ParserCYK(GRAMATICA)
        self.assertEqual(p.analizar(["ssh", "bloquea"]), [])
        self.assertEqual(p.analizar(["bloquea", "palabra_desconocida"]), [])
        self.assertEqual(p.analizar([]), [])

    def test_etiquetas_unk_del_hmm(self):
        p = ParserCYK(GRAMATICA)
        arboles = p.analizar(["deniega", "ssh"], etiquetas_unk={0: "VACC"})
        self.assertEqual(a_corchetes(arboles[0]), "(S (SV (VACC deniega) (SN (SERV ssh))))")

    def test_parametros_invalidos(self):
        p = ParserCYK(GRAMATICA)
        with self.assertRaises(ValueError):
            p.analizar(["bloquea", "ssh"], k=0)
        with self.assertRaises(ValueError):
            p.analizar(["bloquea", "ssh"], tokens=tokens_fixture(["bloquea"]))
        with self.assertRaises(ValueError):
            p.analizar(["bloquea", "<IP>"], tokens=tokens_fixture(["bloquea", "ssh"]))

    def test_regresion_bucle_infinito_en_clausura_unaria(self):
        # S tiene ya su mejor derivación binaria y le llega una unaria peor (vía C).
        g = gramatica_desde_texto('S -> A B [0.9]\nS -> C [0.1]\nC -> A B [1.0]\nA -> "a" [1.0]\nB -> "b" [1.0]\n')
        p = ParserCYK(g)
        uno = con_limite_de_tiempo(lambda: p.analizar(["a", "b"], k=1))
        self.assertEqual([round(math.exp(a.logprob), 9) for a in uno], [0.9])
        dos = con_limite_de_tiempo(lambda: p.analizar(["a", "b"], k=2))
        self.assertEqual([round(math.exp(a.logprob), 9) for a in dos], [0.9, 0.1])

    def test_regresion_punteros_unarios_desactualizados(self):
        # B recibe una derivación mejor (vía Z) después de que A ya usó la primera.
        g = gramatica_desde_texto('S -> A [1.0]\nA -> B [1.0]\nB -> "x" [0.1]\nB -> Z [0.9]\nZ -> "x" [1.0]\n')
        arboles = ParserCYK(g).analizar(["x"], k=2)
        self.assertEqual(
            [(round(math.exp(a.logprob), 9), a_corchetes(a)) for a in arboles],
            [(0.9, "(S (A (B (Z x))))"), (0.1, "(S (A (B x)))")],
        )
        for a in arboles:
            self.assertAlmostEqual(a.logprob, logprob_por_reglas(g, a), places=9)

    def test_ciclo_unario_termina(self):
        g = gramatica_desde_texto('S -> A [1.0]\nA -> B [0.5]\nA -> "x" [0.5]\nB -> A [1.0]\n')
        arboles = con_limite_de_tiempo(lambda: ParserCYK(g).analizar(["x"], k=3))
        self.assertEqual([round(math.exp(a.logprob), 9) for a in arboles], [0.5, 0.25, 0.125])


if __name__ == "__main__":
    unittest.main()
