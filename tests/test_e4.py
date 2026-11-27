"""Pruebas del código del escenario E4 (experimentos/e4_clasificacion_etiquetado.py). Dueño: rol 4."""
import contextlib
import csv
import importlib.util
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import FIXTURES, RAIZ, tokens_fixture
from ordenred.corpus import cargar_jsonl, guardar_jsonl
from ordenred.hmm import HMM, TERMINALES_ESPECIALES
from ordenred.semantica import ACCIONES
from ordenred.tokenizador import tokenizar

_spec = importlib.util.spec_from_file_location(
    "e4_clasificacion_etiquetado", RAIZ / "experimentos" / "e4_clasificacion_etiquetado.py"
)
e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e4)

MINI = cargar_jsonl(FIXTURES / "mini_corpus.jsonl")


def _tokenizar_es_stub() -> bool:
    try:
        tokenizar("bloquea ssh")
    except NotImplementedError:
        return True
    return False


def registro(id_, tokens, etiquetas, intencion="BLOQUEAR"):
    return {"id": id_, "orden": " ".join(tokens), "tokens": tokens, "etiquetas": etiquetas, "intencion": intencion}


class TestSubmuestra(unittest.TestCase):
    def test_tamano(self):
        regs = [{"id": i} for i in range(40)]
        for fraccion, esperado in ((0.1, 4), (0.25, 10), (0.5, 20), (1.0, 40), (0.01, 1)):
            self.assertEqual(len(e4.submuestra(regs, fraccion, 3)), esperado)

    def test_determinista_y_sin_repeticion(self):
        regs = [{"id": i} for i in range(40)]
        a = e4.submuestra(regs, 0.25, 7)
        self.assertEqual(a, e4.submuestra(regs, 0.25, 7))
        self.assertEqual(len({r["id"] for r in a}), len(a))
        self.assertNotEqual(a, e4.submuestra(regs, 0.25, 8))

    def test_anidadas_por_semilla(self):
        regs = [{"id": i} for i in range(40)]
        for semilla in range(5):
            previos: set[int] = set()
            for fraccion in (0.1, 0.25, 0.5, 1.0):
                ids = {r["id"] for r in e4.submuestra(regs, fraccion, semilla)}
                self.assertTrue(previos <= ids, (semilla, fraccion))
                previos = ids
            self.assertEqual(previos, set(range(40)))

    def test_conserva_el_orden_del_corpus(self):
        regs = [{"id": i} for i in range(40)]
        ids = [r["id"] for r in e4.submuestra(regs, 0.5, 2)]
        self.assertEqual(ids, sorted(ids))

    def test_fraccion_invalida(self):
        with self.assertRaises(ValueError):
            e4.submuestra([{"id": 0}], 0.0, 0)


class TestMatrizConfusion(unittest.TestCase):
    def test_caso_a_mano(self):
        reales = ["BLOQUEAR", "BLOQUEAR", "PERMITIR", "PERMITIR", "PERMITIR"]
        predichas = ["BLOQUEAR", "PERMITIR", "PERMITIR", "PERMITIR", "BLOQUEAR"]
        m = e4.matriz_confusion(reales, predichas, ACCIONES)
        self.assertEqual(set(m), set(ACCIONES))
        self.assertTrue(all(set(fila) == set(ACCIONES) for fila in m.values()))
        self.assertEqual(m["BLOQUEAR"]["BLOQUEAR"], 1)
        self.assertEqual(m["BLOQUEAR"]["PERMITIR"], 1)
        self.assertEqual(m["PERMITIR"]["PERMITIR"], 2)
        self.assertEqual(m["PERMITIR"]["BLOQUEAR"], 1)
        self.assertEqual(sum(v for fila in m.values() for v in fila.values()), 5)
        self.assertEqual(sum(m["LISTAR"].values()), 0)

    def test_clase_desconocida(self):
        with self.assertRaises(ValueError):
            e4.matriz_confusion(["BLOQUEAR"], ["OTRA"], ACCIONES)


class TestRetenerVocabulario(unittest.TestCase):
    def test_oraciones_sin_retenidas_y_sin_especiales(self):
        for semilla in range(10):
            train, retenidas = e4.retener_vocabulario(MINI, 0.1, semilla, "fixtures")
            self.assertEqual(len(retenidas), 1)  # round(0.1 · 12 tipos)
            self.assertFalse(retenidas & TERMINALES_ESPECIALES)
            self.assertTrue(all(not w.startswith("<") for w in retenidas))
            for r in train:
                self.assertFalse(retenidas & set(e4.terminales_registro(r, "fixtures")))
            # Se quitan exactamente las oraciones que contienen alguna retenida.
            quitadas = [r for r in MINI if r not in train]
            for r in quitadas:
                self.assertTrue(retenidas & set(e4.terminales_registro(r, "fixtures")))

    def test_determinista(self):
        self.assertEqual(
            e4.retener_vocabulario(MINI, 0.2, 5, "fixtures"), e4.retener_vocabulario(MINI, 0.2, 5, "fixtures")
        )

    def test_retener_todo_falla(self):
        with self.assertRaises(ValueError):
            e4.retener_vocabulario(MINI, 1.0, 0, "fixtures")

    def test_conserva_etiquetas_y_respeta_max_fraccion(self):
        etiquetas_train = {t for r in MINI for t in r["etiquetas"]}
        oraciones = [set(e4.terminales_registro(r, "fixtures")) for r in MINI]
        # En el mini corpus solo «a», «acceso» y «http» se pueden retener: las demás son la única palabra
        # de su etiqueta (desde, por, puerto, el), salen en más del 30 % de las oraciones (bloquea, ssh…)
        # o su oración es la única con PPUERTO/NPUERTO/NUMERO (trafico, hacia).
        vistas = set()
        for semilla in range(30):
            train, retenidas = e4.retener_vocabulario(MINI, 0.1, semilla, "fixtures")
            self.assertEqual({t for r in train for t in r["etiquetas"]}, etiquetas_train)
            for w in retenidas:
                self.assertLessEqual(sum(w in o for o in oraciones), e4.MAX_FRACCION_ORACIONES * len(MINI))
            vistas |= retenidas
        self.assertEqual(vistas, {"a", "acceso", "http"})

    def test_palabra_unica_de_su_etiqueta_no_se_retiene(self):
        # «x» es la única palabra con etiqueta X; las demás salen en todas las oraciones.
        train = [registro(f"t{i}", ["bloquea", "ssh"], ["V", "N"]) for i in range(9)]
        train.append(registro("tx", ["bloquea", "ssh", "x"], ["V", "N", "X"]))
        with self.assertRaisesRegex(ValueError, "etiqueta entera"):
            e4.retener_vocabulario(train, 0.5, 0, "fixtures")

    def test_proporcion_cero(self):
        train, retenidas = e4.retener_vocabulario(MINI, 0.0, 0, "fixtures")
        self.assertEqual((train, retenidas), (MINI, set()))


class TestExactitudHMM(unittest.TestCase):
    def test_caso_a_mano(self):
        # Raras (frecuencia 1, se entrenan como firma): cosa, casa → <UNK-a> con D; tcp → <UNK> con N.
        # Tras V siempre viene N.
        train = [registro(f"t{i}", ["el", "el", "bloquea", "ssh"], ["D", "D", "V", "N"]) for i in range(2)]
        train.append(registro("t2", ["cosa", "casa", "bloquea", "tcp"], ["D", "D", "V", "N"]))
        hmm = e4.entrenar_hmm(train, "fixtures")
        # Base de raras: D tiene 2 emisiones de firma y N 1 → D.
        self.assertEqual(e4.etiqueta_base_raras(hmm), "D")
        self.assertEqual(e4.n_raras_train(hmm), 3)
        # Base de firma sobre hmm.observacion: «ftp» → <UNK> (visto con N) → N; «lista» → <UNK-a>
        # (visto con D) → D; «1x» → <UNK-DIG> no visto → el HMM lo ve como <UNK> → N.
        self.assertEqual(e4.etiqueta_base_firma(hmm, "ftp", "Z"), "N")
        self.assertEqual(e4.etiqueta_base_firma(hmm, "lista", "Z"), "D")
        self.assertEqual(e4.etiqueta_base_firma(hmm, "1x", "Z"), "N")
        evaluacion = [
            registro("e1", ["el", "bloquea", "ftp"], ["D", "V", "N"]),
            registro("e2", ["el", "bloquea", "lista"], ["D", "V", "N"]),
            {**registro("e3", ["bloquea", "eso"], None), "fuente": "humano"},  # fuera de cobertura: se descarta
        ]
        # Viterbi a mano para «lista» (<UNK-a>) tras V, con α = 1, C(D)=6, C(N)=3, C(V)=3, T=3,
        # |V′| = 5 (el, bloquea, ssh, <UNK-a>, <UNK>):
        #   N: A(V,N)·B_N(<UNK-a>)·fin_N = 4/7 · 1/8 · 4/7 ≈ 0.041
        #   D: 1/7 · 3/11 · 1/10 ≈ 0.0039;   V: 1/7 · 1/8 · 1/7 ≈ 0.0026   → gana N por el contexto,
        # aunque la firma sola diría D.
        self.assertEqual(hmm.viterbi(["el", "bloquea", "lista"])[0], ["D", "V", "N"])
        self.assertEqual(hmm.viterbi(["el", "bloquea", "ftp"])[0], ["D", "V", "N"])
        res = e4.exactitud_hmm(hmm, evaluacion, "fixtures", {"lista"})
        self.assertEqual(res["n_oraciones"], 2)
        self.assertEqual(res["n_palabras"], 6)
        self.assertEqual(res["exactitud"], 1.0)
        self.assertEqual((res["n_conocidas"], res["exactitud_conocidas"]), (4, 1.0))
        self.assertEqual((res["n_desconocidas"], res["exactitud_desconocidas"]), (2, 1.0))
        self.assertEqual(res["etiqueta_base"], "D")
        self.assertEqual(res["exactitud_desconocidas_base"], 0.0)  # D en ftp y en lista: ambas son N
        self.assertEqual(res["exactitud_base_firma"], 0.5)  # ftp → N (acierta), lista → D (falla)
        self.assertEqual((res["n_retenidas_eval"], res["exactitud_retenidas"]), (1, 1.0))
        self.assertEqual(res["n_raras_train"], 3)

    def test_base_sin_raras_usa_la_etiqueta_con_mas_tipos_de_palabra(self):
        # Sin raras (todo aparece 2 veces). Conteos: D = 6 (la global), N = 4, X = 6, V = 2.
        # Tipos de palabra conocidos: D {el} = 1, V {bloquea} = 1, N {ssh, http} = 2; X tiene 3 tipos pero
        # son terminales especiales (<IP>, <CIDR>, <NUM>) y no cuentan → respaldo N.
        formas = ["el", "el", "el", "bloquea", "ssh", "http", "10.0.0.1", "10.0.0.0/8", "22"]
        etiquetas = ["D", "D", "D", "V", "N", "N", "X", "X", "X"]
        train = [registro(f"t{i}", formas, etiquetas) for i in range(2)]
        hmm = e4.entrenar_hmm(train, "fixtures")
        self.assertEqual(hmm.firmas, set())
        self.assertEqual(e4.n_raras_train(hmm), 0)
        self.assertEqual(e4.etiqueta_base_raras(hmm), "N")
        # <UNK> no tiene conteos: la base de firma usa el respaldo y coincide con la de raras.
        self.assertEqual(e4.etiqueta_base_firma(hmm, "ftp", "N"), "N")
        res = e4.exactitud_hmm(hmm, [registro("e", ["bloquea", "ftp"], ["V", "N"])], "fixtures")
        self.assertEqual((res["n_raras_train"], res["etiqueta_base"]), (0, "N"))
        self.assertEqual(res["n_desconocidas"], 1)
        self.assertEqual(res["exactitud_desconocidas_base"], res["exactitud_base_firma"])
        self.assertEqual(res["exactitud_desconocidas_base"], 1.0)

    def test_sin_desconocidas_da_nan(self):
        train = [registro(f"t{i}", ["bloquea", "ssh"], ["V", "N"]) for i in range(2)]
        hmm = e4.entrenar_hmm(train, "fixtures")
        res = e4.exactitud_hmm(hmm, train, "fixtures")
        self.assertEqual(res["n_desconocidas"], 0)
        for clave in ("exactitud_desconocidas", "exactitud_desconocidas_base", "exactitud_base_firma",
                      "exactitud_retenidas"):
            self.assertTrue(math.isnan(res[clave]), clave)


class TestCurvaYCorrer(unittest.TestCase):
    def test_curva_aprendizaje(self):
        filas = e4.curva_aprendizaje(MINI, MINI, (0.1, 1.0), [0, 1], "fixtures")
        self.assertEqual(len(filas), 2 * 2 * 2)
        self.assertEqual({f["modelo"] for f in filas}, {"nb", "rl"})
        self.assertEqual({f["n_train"] for f in filas if f["fraccion"] == 0.1}, {1})
        self.assertEqual({f["n_train"] for f in filas if f["fraccion"] == 1.0}, {5})
        self.assertTrue(all(0.0 <= f["exactitud"] <= 1.0 for f in filas))
        # Diagnóstico de la RL: solo en sus filas; con una sola clase (10 % = 1 orden) no hay pérdida.
        for f in filas:
            if f["modelo"] == "nb":
                self.assertEqual((f["epocas"], f["perdida_final"], f["convergio"]), (None, None, None))
            elif f["fraccion"] == 1.0:
                self.assertEqual(f["epocas"], e4.EPOCAS_RL)
                self.assertTrue(math.isfinite(f["perdida_final"]) and f["perdida_final"] > 0)
                self.assertIn(f["convergio"], (True, False))
            else:
                self.assertTrue(math.isnan(f["perdida_final"]))
                self.assertIsNone(f["convergio"])

    def test_epocas_rl_y_perdida_final(self):
        pocas = e4.curva_aprendizaje(MINI, MINI, (1.0,), [0], "fixtures", epocas_rl=5)
        muchas = e4.curva_aprendizaje(MINI, MINI, (1.0,), [0], "fixtures", epocas_rl=300)
        rl_pocas = next(f for f in pocas if f["modelo"] == "rl")
        rl_muchas = next(f for f in muchas if f["modelo"] == "rl")
        self.assertEqual((rl_pocas["epocas"], rl_muchas["epocas"]), (5, 300))
        self.assertLess(rl_muchas["perdida_final"], rl_pocas["perdida_final"])
        self.assertIsNone(rl_pocas["convergio"])  # 5 épocas ≤ ventana de 10: no se puede juzgar
        with self.assertRaises(ValueError):
            e4.curva_aprendizaje(MINI, MINI, (1.0,), [0], "fixtures", epocas_rl=0)

    def test_convergio(self):
        v = e4.VENTANA_CONVERGENCIA
        self.assertIsNone(e4.convergio([1.0] * v))
        self.assertFalse(e4.convergio([2.0] + [1.0] * v))  # L[-1-v] = 2.0: cae a la mitad en la ventana
        self.assertTrue(e4.convergio([2.0] + [1.0] * (v + 1)))  # la caída quedó fuera de la ventana
        self.assertTrue(e4.convergio([1.0] * (v + 1)))
        # Pérdida grande: umbral relativo 1e-3 · L. Baja 0,1 en la ventana → no; baja 1e-4 (< 1e-3·0,9999) → sí.
        self.assertFalse(e4.convergio([1.0 - 0.01 * k for k in range(v + 1)]))
        self.assertTrue(e4.convergio([1.0 - 1e-5 * k for k in range(v + 1)]))
        # Pérdida cercana a 0 (datos separables): 0,008 que baja 1e-6 por época. Mejora 1e-5 < TOL_ABS = 1e-4,
        # aunque en términos relativos (1e-5 / 0,008 ≈ 1,3e-3) superaría TOL_REL → converge.
        self.assertTrue(e4.convergio([0.008 + 1e-6 * (v - k) for k in range(v + 1)]))
        # Cerca de 0 pero aún bajando 1e-3 por época: mejora 1e-2 > max(1e-3·0,01, 1e-4) → no converge.
        self.assertFalse(e4.convergio([0.02 - 1e-3 * k for k in range(v + 1)]))
        # Una pérdida que sube mucho tampoco cuenta como convergida.
        self.assertFalse(e4.convergio([1.0 + 0.01 * k for k in range(v + 1)]))

    def test_correr_y_cli_producen_las_tres_tablas(self):
        rutas = {"train": FIXTURES / "mini_corpus.jsonl", "eval": FIXTURES / "mini_corpus.jsonl"}
        filas = e4.correr(0, rutas, "fixtures")
        self.assertEqual({f["tabla"] for f in filas}, set(e4.TABLAS))
        self.assertEqual(sum(f["tabla"] == "curvas" for f in filas), 4 * 2)
        self.assertEqual(sum(f["tabla"] == "confusion" for f in filas), 2 * len(ACCIONES) ** 2)
        self.assertEqual({f["condicion"] for f in filas if f["tabla"] == "hmm_unk"}, {"completo", "retenido"})

        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                codigo = e4.main(["--fixtures", "--semillas", "2", "--salida", tmp])
            self.assertEqual(codigo, 0)
            for tabla, nombre in e4.ARCHIVOS.items():
                with open(Path(tmp) / nombre, encoding="utf-8", newline="") as f:
                    filas_csv = list(csv.DictReader(f))
                self.assertTrue(filas_csv, tabla)
            with open(Path(tmp) / "e4_curvas.csv", encoding="utf-8", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2 * 4 * 2)

    def test_cli_fixtures_no_admite_train_eval(self):
        for extra in (["--train", "x.jsonl"], ["--eval", "x.jsonl"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                e4.main(["--fixtures", *extra])
            self.assertEqual(cm.exception.code, 2)

    def test_cli_modo_tokenizar_con_tokenizador_simulado(self):
        def tokenizar_simulado(texto):  # separa por espacios y tipa como tokens_fixture
            return tokens_fixture(texto.split())

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            guardar_jsonl(MINI, corpus)
            salida = Path(tmp) / "salida"
            with mock.patch("ordenred.tokenizador.tokenizar", tokenizar_simulado), \
                    contextlib.redirect_stdout(io.StringIO()):
                codigo = e4.main(["--train", str(corpus), "--eval", str(corpus), "--semillas", "1",
                                  "--salida", str(salida)])
            self.assertEqual(codigo, 0)
            self.assertEqual({p.name for p in salida.iterdir()}, set(e4.ARCHIVOS.values()))

            # Si las formas del tokenizador no coinciden con `tokens`, error claro y código 1.
            malo = [{**MINI[0], "tokens": ["otra", "cosa"]}, *MINI[1:]]
            guardar_jsonl(malo, corpus)
            with mock.patch("ordenred.tokenizador.tokenizar", tokenizar_simulado), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                codigo = e4.main(["--train", str(corpus), "--eval", str(corpus), "--semillas", "1",
                                  "--salida", str(salida)])
            self.assertEqual(codigo, 1)
            self.assertIn("mini-001", err.getvalue())

    def test_archivo_inexistente(self):
        with self.assertRaisesRegex(FileNotFoundError, "rol 5"):
            e4.correr(0, {"train": "/no/existe.jsonl", "eval": "/no/existe.jsonl"}, "fixtures")


class TestTerminalesRegistro(unittest.TestCase):
    def test_fixtures(self):
        self.assertEqual(
            e4.terminales_registro(MINI[0], "fixtures"), ["bloquea", "el", "ssh", "desde", "<CIDR>"]
        )

    @unittest.skipUnless(_tokenizar_es_stub(), "tokenizar ya está implementado")
    def test_tokenizar_stub_da_runtimeerror(self):
        with self.assertRaisesRegex(RuntimeError, "--fixtures"):
            e4.terminales_registro(MINI[0], "tokenizar")

    def test_modo_desconocido(self):
        with self.assertRaises(ValueError):
            e4.terminales_registro(MINI[0], "otro")


if __name__ == "__main__":
    unittest.main()
