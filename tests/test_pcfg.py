"""Pruebas de pcfg.py (C2, M5a, M6). Dueño: rol 1."""
import math
import tempfile
import unittest
from pathlib import Path

from tests import FIXTURES, tokens_fixture
from ordenred.arbol import desde_corchetes
from ordenred.corpus import cargar_jsonl
from ordenred.cyk import ParserCYK
from ordenred.pcfg import PCFG

MINI = FIXTURES / "mini.pcfg"
CORPUS = cargar_jsonl(FIXTURES / "mini_corpus.jsonl")


def treebank(registros=CORPUS):
    return [desde_corchetes(r["arbol"], tokens_fixture(r["tokens"])) for r in registros]


def prob(g: PCFG, izq: str, *der: str) -> float:
    return next(r.prob for r in g.reglas if (r.izq, r.der) == (izq, der))


class TestLectura(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def escribir(self, texto: str) -> Path:
        ruta = Path(self.dir.name) / "g.pcfg"
        ruta.write_text(texto, encoding="utf-8")
        return ruta

    def test_lee_la_mini_gramatica(self):
        g = PCFG.desde_archivo(MINI)
        self.assertEqual(len(g.reglas), 29)
        self.assertEqual(prob(g, "SV", "VACC", "SN"), 0.6)
        self.assertEqual(prob(g, "DIR", '"<CIDR>"'), 0.5)
        self.assertIn("<cidr>", g.indice_lexico)
        self.assertEqual({r.izq for r in g.indice_unario["SV"]}, {"S"})
        self.assertEqual({r.izq for r in g.indice_binario[("SV", "SP")]}, {"SV"})

    def test_comentario_al_final_de_la_linea(self):
        g = PCFG.desde_archivo(self.escribir('S -> A [1.0]   # raíz\nA -> "a" [1]\n'))
        self.assertEqual(len(g.reglas), 2)

    def test_errores_de_formato(self):
        malos = {
            "no suma 1": 'S -> A [0.7]\nA -> "a" [1.0]\n',
            "ternaria": 'S -> A A A [1.0]\nA -> "a" [1.0]\n',
            "terminal mezclado": 'S -> "a" A [1.0]\nA -> "a" [1.0]\n',
            "sin probabilidad": 'S -> A\nA -> "a" [1.0]\n',
            "probabilidad no numérica": 'S -> A [uno]\nA -> "a" [1.0]\n',
            "probabilidad cero": 'S -> A [1.0]\nS -> B [0.0]\nA -> "a" [1.0]\n',
            "terminal sin comillas": "S -> A [1.0]\nA -> bloquea [1.0]\n",
            "duplicada": 'S -> A [0.5]\nS -> A [0.5]\nA -> "a" [1.0]\n',
            "sin símbolo inicial": 'X -> A [1.0]\nA -> "a" [1.0]\n',
            "sin flecha": 'S A [1.0]\nA -> "a" [1.0]\n',
        }
        for nombre, texto in malos.items():
            with self.subTest(nombre), self.assertRaises(ValueError):
                PCFG.desde_archivo(self.escribir(texto))
        with self.assertRaises(FileNotFoundError):
            PCFG.desde_archivo(Path(self.dir.name) / "no_existe.pcfg")

    def test_guardar_y_releer_conserva_las_probabilidades(self):
        g = PCFG.desde_archivo(MINI)
        g.estimar_desde_treebank(treebank(), suavizado=1.0)  # probabilidades con muchos decimales
        ruta = Path(self.dir.name) / "sub" / "rt.pcfg"
        g.guardar(ruta)
        g2 = PCFG.desde_archivo(ruta)
        self.assertEqual([(r.izq, r.der) for r in g2.reglas], [(r.izq, r.der) for r in g.reglas])
        for a, b in zip(g.reglas, g2.reglas):
            self.assertAlmostEqual(a.prob, b.prob, places=10)


class TestEstimacion(unittest.TestCase):
    def test_maxima_verosimilitud(self):
        g = PCFG.desde_archivo(MINI)
        g.estimar_desde_treebank(treebank())
        # 5 órdenes: 3 «bloquea» y 2 «permite».
        self.assertAlmostEqual(prob(g, "VACC", '"bloquea"'), 3 / 5)
        self.assertAlmostEqual(prob(g, "VACC", '"permite"'), 2 / 5)
        self.assertAlmostEqual(prob(g, "S", "SV"), 1.0)
        # «la» no aparece en el treebank: con α = 0 la MLE la descarta.
        self.assertNotIn("la", g.indice_lexico)
        # Los especiales se cuentan como terminales <IP>/<CIDR>, no como la dirección concreta.
        self.assertIn("<cidr>", g.indice_lexico)
        self.assertNotIn("10.0.0.0/8", g.indice_lexico)

    def test_suavizado_conserva_las_reglas_no_vistas(self):
        g = PCFG.desde_archivo(MINI)
        g.estimar_desde_treebank(treebank(), suavizado=1.0)
        self.assertEqual(len(g.reglas), 29)
        # DET: «el» 4 veces, «la» 0 veces → (4+1)/(4+2) y (0+1)/(4+2).
        self.assertAlmostEqual(prob(g, "DET", '"el"'), 5 / 6)
        self.assertAlmostEqual(prob(g, "DET", '"la"'), 1 / 6)
        self.assertEqual(len(ParserCYK(g).analizar(["bloquea", "la", "ssh"])), 1)

    def test_no_terminales_ausentes_conservan_sus_probabilidades(self):
        g = PCFG.desde_archivo(MINI)
        g.estimar_desde_treebank(treebank([r for r in CORPUS if r["orden"] == "bloquea ssh"]))
        self.assertAlmostEqual(prob(g, "SP", "PORIG", "SN"), 0.4)  # SP no aparece en «bloquea ssh»
        self.assertAlmostEqual(prob(g, "SN", "SERV"), 1.0)  # SN sí aparece, solo como SERV

    def test_reglas_nuevas_del_treebank_se_agregan(self):
        g = PCFG.desde_archivo(MINI)
        a = desde_corchetes("(S (SV (VACC deniega) (SN (SERV ssh))))", tokens_fixture(["deniega", "ssh"]))
        g.estimar_desde_treebank([a], suavizado=1.0)
        self.assertIn("deniega", g.indice_lexico)
        g.validar_consistencia()

    def test_treebank_vacio_no_cambia_nada(self):
        g = PCFG.desde_archivo(MINI)
        antes = [(r.izq, r.der, r.prob) for r in g.reglas]
        g.estimar_desde_treebank([])
        self.assertEqual([(r.izq, r.der, r.prob) for r in g.reglas], antes)

    def test_desde_gramatica_vacia(self):
        g = PCFG()
        g.estimar_desde_treebank(treebank())
        g.validar_consistencia()
        self.assertAlmostEqual(math.fsum(r.prob for r in g.reglas if r.izq == "SN"), 1.0)

    def test_suavizado_negativo(self):
        with self.assertRaises(ValueError):
            PCFG.desde_archivo(MINI).estimar_desde_treebank(treebank(), suavizado=-1)


if __name__ == "__main__":
    unittest.main()
