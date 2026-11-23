"""Pruebas de arbol.py (C3). Dueño: rol 1."""
import unittest

from tests import FIXTURES
from ordenred.arbol import Arbol, a_corchetes, desde_corchetes, hojas, preterminales, reglas
from ordenred.corpus import cargar_jsonl
from ordenred.tokenizador import Token

CORPUS = cargar_jsonl(FIXTURES / "mini_corpus.jsonl")


class TestArbol(unittest.TestCase):
    def test_ida_y_vuelta_con_el_corpus(self):
        for r in CORPUS:
            with self.subTest(orden=r["orden"]):
                self.assertEqual(a_corchetes(desde_corchetes(r["arbol"])), r["arbol"])

    def test_preterminales_y_hojas(self):
        for r in CORPUS:
            with self.subTest(orden=r["orden"]):
                a = desde_corchetes(r["arbol"])
                self.assertEqual(preterminales(a), r["etiquetas"])
                self.assertEqual([t.forma for t in hojas(a)], r["tokens"])

    def test_espacios_y_saltos_de_linea_no_importan(self):
        a = desde_corchetes("(S\n  (SV (VACC bloquea)\n      (SN (SERV ssh))))")
        self.assertEqual(a_corchetes(a), "(S (SV (VACC bloquea) (SN (SERV ssh))))")

    def test_reglas_con_tokens_especiales(self):
        toks = [Token("bloquea", "bloquea", "PALABRA"), Token("10.0.0.5", "10.0.0.5", "IP")]
        a = desde_corchetes("(S (SV (VACC bloquea) (SN (DIR 10.0.0.5))))", toks)
        self.assertEqual(
            reglas(a),
            [("S", ("SV",)), ("SV", ("VACC", "SN")), ("VACC", ('"bloquea"',)),
             ("SN", ("DIR",)), ("DIR", ('"<IP>"',))],
        )
        self.assertIs(hojas(a)[1], toks[1])

    def test_tokens_que_no_coinciden(self):
        toks = [Token("permite", "permite", "PALABRA"), Token("ssh", "ssh", "PALABRA")]
        with self.assertRaises(ValueError):
            desde_corchetes("(S (SV (VACC bloquea) (SN (SERV ssh))))", toks)
        with self.assertRaises(ValueError):
            desde_corchetes("(S (SV (VACC bloquea) (SN (SERV ssh))))", toks[:1])

    def test_arboles_mal_formados(self):
        for malo in ["", "(S", "(S (VACC bloquea)", "(S (VACC bloquea)) sobra", "((S x))", "(S )", "S x"]:
            with self.subTest(malo=malo), self.assertRaises(ValueError):
                desde_corchetes(malo)

    def test_nodos_inconsistentes_no_se_serializan(self):
        with self.assertRaises(ValueError):
            a_corchetes(Arbol("S"))
        t = Token("ssh", "ssh", "PALABRA")
        with self.assertRaises(ValueError):
            a_corchetes(Arbol("SERV", [Arbol("X", token=t)], token=t))


if __name__ == "__main__":
    unittest.main()
