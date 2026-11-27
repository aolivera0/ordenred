"""Pruebas de los contratos del esqueleto (C1–C8) y de los fixtures. Dueño: rol 1.

Cuando un rol implemente su módulo, debe QUITAR su entrada de PENDIENTES; esta prueba
fallará a propósito si un método ya implementado sigue listado, para que la lista no envejezca.
"""

import json
import re
import unittest
from dataclasses import fields
from tests import FIXTURES
from ordenred import (
    cli,
    clasificadores,
    corrector,
    cyk,
    generador_reglas,
    hmm,
    ngramas,
    objetos,
    pcfg,
    pipeline,
    semantica,
    tokenizador,
)
from ordenred.corpus import cargar_jsonl, validar_registro
from ordenred.semantica import FormaLogica
from ordenred.tokenizador import Token

LM = ngramas.ModeloNGramas(2)

# (nombre, función que llama al stub)
PENDIENTES = [
    ("tokenizar (rol 5)", lambda: tokenizador.tokenizar("bloquea ssh")),
    ("ModeloNGramas.entrenar (rol 3)", lambda: LM.entrenar([["a"]])),
    ("ModeloNGramas.ajustar_lambdas (rol 3)", lambda: LM.ajustar_lambdas([["a"]])),
    ("ModeloNGramas.logprob (rol 3)", lambda: LM.logprob(["a"])),
    ("ModeloNGramas.perplejidad (rol 3)", lambda: LM.perplejidad([["a"]])),
    ("Corrector.corregir (rol 3)", lambda: corrector.Corrector(set(), LM, LM).corregir([])),
    ("NaiveBayes.entrenar (rol 4)", lambda: clasificadores.NaiveBayes().entrenar([], [])),
    ("NaiveBayes.predecir (rol 4)", lambda: clasificadores.NaiveBayes().predecir([])),
    (
        "RegresionLogistica.entrenar (rol 4)",
        lambda: clasificadores.RegresionLogistica().entrenar([], []),
    ),
    ("RegresionLogistica.predecir (rol 4)", lambda: clasificadores.RegresionLogistica().predecir([])),
    ("HMM.entrenar (rol 4)", lambda: hmm.HMM().entrenar([])),
    ("HMM.viterbi (rol 4)", lambda: hmm.HMM().viterbi([])),
    # --- ROL 1: IMPLEMENTADO CON ÉXITO POR SERGIO SIABATTO (CONTRATOS EN VERDE) ---
    # ("PCFG.desde_archivo (rol 1)", lambda: pcfg.PCFG.desde_archivo(str(FIXTURES / "mini.pcfg"))),
    # ("PCFG.estimar_desde_treebank (rol 1)", lambda: pcfg.PCFG().estimar_desde_treebank([])),
    # ("PCFG.guardar (rol 1)", lambda: pcfg.PCFG().guardar("/dev/null")),
    # ("ParserCYK.analizar (rol 1)", lambda: cyk.ParserCYK(pcfg.PCFG()).analizar(["bloquea"])),
    (
        "TablaObjetos.desde_archivo (rol 2)",
        lambda: objetos.TablaObjetos.desde_archivo(str(FIXTURES / "mini_objetos.json")),
    ),
    ("TablaObjetos.servicio (rol 2)", lambda: objetos.TablaObjetos().servicio("ssh")),
    ("TablaObjetos.red (rol 2)", lambda: objetos.TablaObjetos().red("invitados")),
    ("FormaLogica.a_logica (rol 2)", lambda: FormaLogica("BLOQUEAR").a_logica()),
    (
        "Interprete.interpretar (rol 2)",
        lambda: semantica.Interprete(objetos.TablaObjetos()).interpretar(None),
    ),
    ("a_iptables (rol 2)", lambda: generador_reglas.a_iptables([])),
    ("a_acl_cisco (rol 2)", lambda: generador_reglas.a_acl_cisco([])),
    ("Pipeline.traducir (rol 6)", lambda: pipeline.Pipeline().traducir("bloquea ssh")),
]


class TestStubsPendientes(unittest.TestCase):
    def test_los_stubs_listados_siguen_sin_implementar(self):
        for nombre, llamada in PENDIENTES:
            with (
                self.subTest(nombre),
                self.assertRaises(
                    NotImplementedError, msg=f"{nombre} ya está implementado: quítelo de PENDIENTES"
                ),
            ):
                llamada()


class TestContratos(unittest.TestCase):
    def test_c1_token(self):
        self.assertEqual([f.name for f in fields(Token)], ["texto", "forma", "tipo"])
        self.assertEqual(Token("Bloquea", "bloquea", "PALABRA").terminal, "bloquea")
        self.assertEqual(Token("10.0.0.0/8", "10.0.0.0/8", "CIDR").terminal, "<CIDR>")
        with self.assertRaises(ValueError):
            Token("x", "x", "OTRO")

    def test_c4_forma_logica(self):
        fl = FormaLogica("BLOQUEAR", "tcp", origen="10.0.0.0/8", puerto_destino=[22, 22])
        self.assertEqual(fl.puerto_destino, (22, 22))
        d = fl.a_dict()
        self.assertEqual(
            d,
            {
                "accion": "BLOQUEAR",
                "protocolo": "tcp",
                "origen": "10.0.0.0/8",
                "puerto_destino": [22, 22],
                "cadena": "INPUT",
            },
        )
        self.assertEqual(FormaLogica.desde_dict(json.loads(json.dumps(d))), fl)

        for malo in [
            dict(accion="BORRAR"),
            dict(accion="BLOQUEAR", protocolo="sctp"),
            dict(accion="BLOQUEAR", cadena="PREROUTING"),
            dict(accion="BLOQUEAR", puerto_destino=(80, 22)),
            dict(accion="BLOQUEAR", puerto_origen=(1, 70000)),
        ]:
            with self.subTest(malo=malo), self.assertRaises(ValueError):
                FormaLogica(**malo)

        with self.assertRaises(ValueError):
            FormaLogica.desde_dict({"accion": "BLOQUEAR", "puerto": 22})

    def test_c6_validacion_de_parametros(self):
        with self.assertRaises(ValueError):
            ngramas.ModeloNGramas(0)
        with self.assertRaises(ValueError):
            ngramas.ModeloNGramas(2, suavizado="kneser-ney")
        with self.assertRaises(ValueError):
            ngramas.ModeloNGramas(2, unidad="silaba")

    def test_c7_pipeline(self):
        self.assertEqual(pipeline.Pipeline({"k": 5}).config["k"], 5)
        with self.assertRaises(ValueError):
            pipeline.Pipeline({"opcion_inexistente": 1})
        r = pipeline.Resultado("bloquea ssh")
        self.assertEqual((r.tokens, r.arboles, r.iptables), ([], [], []))

    def test_cli_argumentos(self):
        args = cli.construir_parser().parse_args(
            ["traducir", "bloquea ssh", "--formato", "acl", "--k", "2"]
        )
        self.assertEqual(
            (args.comando, args.orden, args.formato, args.k), ("traducir", "bloquea ssh", "acl", 2)
        )


class TestFixtures(unittest.TestCase):
    def setUp(self):
        self.corpus = cargar_jsonl(FIXTURES / "mini_corpus.jsonl")
        self.gramatica = {}
        patron = re.compile(r'^(\S+)\s*->\s*(.+?)\s*\[([0-9.]+)\]\s*$')
        for linea in (FIXTURES / "mini.pcfg").read_text(encoding="utf-8").splitlines():
            linea = linea.split("#")[0].strip()
            if linea:
                a, rhs, p = patron.match(linea).groups()
                self.gramatica.setdefault(a, {})[tuple(rhs.split())] = float(p)

    def test_corpus_cumple_c5(self):
        for r in self.corpus:
            with self.subTest(id=r["id"]):
                self.assertEqual(validar_registro(r), [])

    def test_validador_detecta_errores(self):
        r = dict(self.corpus[0], etiquetas=["X"] * len(self.corpus[0]["tokens"]), intencion="PERMITIR")
        errores = validar_registro(r)
        self.assertTrue(any("preterminales" in e for e in errores))
        self.assertTrue(any("accion no coincide" in e for e in errores))
        self.assertTrue(validar_registro({"id": "x"}))

    def test_mini_pcfg_suma_uno(self):
        for a, reglas_nt in self.gramatica.items():
            with self.subTest(no_terminal=a):
                self.assertAlmostEqual(sum(reglas_nt.values()), 1.0, places=9)

    def test_esperado_del_cyk_es_coherente(self):
        esperado = json.loads(
            (FIXTURES / "mini_cyk_esperado.json").read_text(encoding="utf-8")
        )["casos"]
        self.assertEqual([c["orden"] for c in esperado], [r["orden"] for r in self.corpus])
        for c, r in zip(esperado, self.corpus):
            with self.subTest(orden=c["orden"]):
                self.assertEqual(len(c["probs_ordenadas"]), c["n_arboles"])
                self.assertEqual(r["ambigua"], c["n_arboles"] > 1)


if __name__ == "__main__":
    unittest.main()
