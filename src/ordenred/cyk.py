"""
M5b · Analizador CYK probabilístico con k mejores árboles (contrato C6). Dueño: rol 1 (Sergio Nicolás Siabatto Cleves).
AIMA §24.3, fig. 24.5. Tabla NumPy en log-probabilidades, clausura unaria y punteros de retroceso.
"""

from __future__ import annotations

import math
from .arbol import Arbol
from .pcfg import PCFG, ReglaPCFG
from .tokenizador import Token


class ParserCYK:
    def __init__(self, pcfg: PCFG) -> None:
        self.pcfg = pcfg

    def analizar(
        self,
        terminales: list[str],
        k: int = 1,
        etiquetas_unk: dict[int, str] | None = None,
    ) -> list[Arbol]:
        """
        Devuelve hasta k árboles con raíz S, ordenados de mayor a menor probabilidad.

        terminales: `Token.terminal` de cada token ("bloquea", "<CIDR>", ...).
        etiquetas_unk: {posición: preterminal} propuesto por el HMM para palabras fuera del léxico.
        Lista vacía si la orden no tiene análisis.
        """
        n = len(terminales)
        if n == 0:
            return []

        # tabla[i][j][no_terminal] -> lista de tuplas (logprob, backpointer)
        tabla: list[list[dict[str, list[tuple[float, tuple]]]]] = [
            [{} for _ in range(n)] for _ in range(n)
        ]

        # --- 1. Caso Base: Longitud 1 (Preterminales) ---
        for i in range(n):
            term = terminales[i]
            term_min = term.lower()

            es_unk = False
            if etiquetas_unk and i in etiquetas_unk:
                categoria_sugerida = etiquetas_unk[i]
                tabla[i][i][categoria_sugerida] = [(0.0, ("terminal", term))]
                es_unk = True

            if not es_unk:
                reglas_lexicas = self.pcfg.indice_lexico.get(term_min, [])
                for regla in reglas_lexicas:
                    if regla.izq not in tabla[i][i]:
                        tabla[i][i][regla.izq] = []
                    tabla[i][i][regla.izq].append((regla.logprob, ("terminal", term)))

            for nt in tabla[i][i]:
                tabla[i][i][nt].sort(key=lambda x: x[0], reverse=True)

            self._aplicar_clausura_unaria(tabla[i][i], limite_k=k)

        # --- 2. Caso Recursivo: Longitudes de span de 2 a N ---
        for longitud in range(2, n + 1):
            for i in range(n - longitud + 1):
                j = i + longitud - 1
                for k_split in range(i, j):
                    celda_izq = tabla[i][k_split]
                    celda_der = tabla[k_split + 1][j]

                    for B, alternativas_B in celda_izq.items():
                        for C, alternativas_C in celda_der.items():
                            reglas_binarias = self.pcfg.indice_binario.get((B, C), [])
                            for regla in reglas_binarias:
                                if regla.izq not in tabla[i][j]:
                                    tabla[i][j][regla.izq] = []

                                for idx_b, (prob_b, _) in enumerate(alternativas_B):
                                    for idx_c, (prob_c, _) in enumerate(alternativas_C):
                                        prob_total = regla.logprob + prob_b + prob_c
                                        bp = ("binaria", k_split, B, idx_b, C, idx_c, regla)
                                        tabla[i][j][regla.izq].append((prob_total, bp))

                # Podar y ordenar las alternativas en la celda
                for nt in list(tabla[i][j].keys()):
                    tabla[i][j][nt].sort(key=lambda x: x[0], reverse=True)
                    tabla[i][j][nt] = tabla[i][j][nt][:k]

                # Aplicar clausura unaria en la celda combinada
                self._aplicar_clausura_unaria(tabla[i][j], limite_k=k)

        # --- 3. Reconstrucción de los k mejores árboles ---
        alternativas_raiz = tabla[0][n - 1].get(self.pcfg.simbolo_inicial, [])
        arboles_resultado: list[Arbol] = []

        for prob_raiz, bp_raiz in alternativas_raiz[:k]:
            arbol_optimo = self._reconstruir_arbol(
                self.pcfg.simbolo_inicial, prob_raiz, bp_raiz, tabla, 0, n - 1
            )
            arboles_resultado.append(arbol_optimo)

        return arboles_resultado

    def _aplicar_clausura_unaria(
        self, celda: dict[str, list[tuple[float, tuple]]], limite_k: int = 1
    ) -> None:
        """Aplica y propaga derivaciones unarias A -> B en la misma celda de la tabla."""
        cambios = True
        while cambios:
            cambios = False
            items_actuales = list(celda.items())
            for B, alternativas_B in items_actuales:
                reglas_unarias = self.pcfg.indice_unario.get(B, [])
                for regla in reglas_unarias:
                    if regla.izq not in celda:
                        celda[regla.izq] = []

                    for idx_b, (prob_b, _) in enumerate(alternativas_B):
                        prob_total = regla.logprob + prob_b
                        bp = ("unaria", B, idx_b, regla)

                        ya_existe = False
                        for p, _ in celda[regla.izq]:
                            if abs(p - prob_total) < 1e-9:
                                ya_existe = True
                                break

                        if not ya_existe:
                            celda[regla.izq].append((prob_total, bp))
                            cambios = True

            for nt in celda:
                celda[nt].sort(key=lambda x: x[0], reverse=True)
                celda[nt] = celda[nt][:limite_k]

    def _reconstruir_arbol(
        self,
        etiqueta: str,
        logprob: float,
        bp: tuple,
        tabla: list[list[dict]],
        i: int,
        j: int,
    ) -> Arbol:
        """Reconstruye recursivamente el objeto Arbol usando los punteros de retroceso."""
        tipo = bp[0]

        if tipo == "terminal":
            texto_terminal = bp[1]
            token_temporal = Token(texto=texto_terminal, forma=texto_terminal, tipo="PALABRA")
            return Arbol(etiqueta=etiqueta, hijos=[], token=token_temporal, logprob=logprob)

        if tipo == "unaria":
            B = bp[1]
            idx_b = bp[2]
            prob_hijo, bp_hijo = tabla[i][j][B][idx_b]
            hijo_arbol = self._reconstruir_arbol(B, prob_hijo, bp_hijo, tabla, i, j)
            return Arbol(etiqueta=etiqueta, hijos=[hijo_arbol], logprob=logprob)

        if tipo == "binaria":
            k_split = bp[1]
            B = bp[2]
            idx_b = bp[3]
            C = bp[4]
            idx_c = bp[5]

            prob_B, bp_B = tabla[i][k_split][B][idx_b]
            prob_C, bp_C = tabla[k_split + 1][j][C][idx_c]

            hijo_izq = self._reconstruir_arbol(B, prob_B, bp_B, tabla, i, k_split)
            hijo_der = self._reconstruir_arbol(C, prob_C, bp_C, tabla, k_split + 1, j)
            return Arbol(etiqueta=etiqueta, hijos=[hijo_izq, hijo_der], logprob=logprob)

        raise ValueError(f"Tipo de backpointer desconocido: {bp}")