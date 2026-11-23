"""Regenera mini_corpus.jsonl y mini_cyk_esperado.json con un enumerador exhaustivo de árboles (independiente del CYK).
Ejecutar desde la raíz del proyecto: python3 tests/fixtures/generar_fixtures.py"""
import json, math, re, sys
from functools import lru_cache
sys.path.insert(0, "src")
from ordenred.arbol import desde_corchetes, reglas, preterminales, hojas
from ordenred.tokenizador import Token
from ordenred.corpus import validar_registro

R = re.compile(r'^(\S+)\s*->\s*(.+?)\s*\[([0-9.]+)\]\s*$')
G = {}
for l in open("tests/fixtures/mini.pcfg"):
    l = l.split("#")[0].strip()
    if not l: continue
    a, rhs, p = R.match(l).groups()
    G.setdefault(a, []).append((tuple(rhs.split()), float(p)))

def tipo(f):
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+/\d+", f): return "CIDR"
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", f): return "IP"
    if re.fullmatch(r"\d+", f): return "NUM"
    return "PALABRA"

def todos(terms):
    n = len(terms)
    @lru_cache(None)
    def arb(A, i, k):  # lista de (prob, corchetes) para A sobre terms[i:k]
        out = []
        for rhs, p in G.get(A, []):
            if len(rhs) == 1 and rhs[0].startswith('"'):
                if k == i + 1 and rhs[0] == f'"{terms[i]}"':
                    out.append((p, f"({A} @{i})"))
            elif len(rhs) == 1:
                out += [(p * q, f"({A} {s})") for q, s in arb(rhs[0], i, k)]
            else:
                B, C = rhs
                for j in range(i + 1, k):
                    for q1, s1 in arb(B, i, j):
                        for q2, s2 in arb(C, j, k):
                            out.append((p * q1 * q2, f"({A} {s1} {s2})"))
        return tuple(out)
    return sorted(arb("S", 0, n), reverse=True)

ORO = [
 ("bloquea el ssh desde 10.0.0.0/8",
  "(S (SV (SV (VACC bloquea) (SN (DET el) (NUC (SERV ssh)))) (SP (PORIG desde) (SN (DIR 10.0.0.0/8)))))",
  "BLOQUEAR", [{"accion":"BLOQUEAR","protocolo":"tcp","origen":"10.0.0.0/8","puerto_destino":[22,22]}],
  ["iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 22 -j DROP"],
  ["access-list 101 deny tcp 10.0.0.0 0.255.255.255 any eq 22"]),
 ("permite el trafico hacia 192.168.1.10 por el puerto 80",
  "(S (SV (SV (SV (VACC permite) (SN (DET el) (NUC (NTRAF trafico)))) (SP (PDEST hacia) (SN (DIR 192.168.1.10)))) (SP (PPUERTO por) (SN (DET el) (NUC (NPUERTO puerto) (NUMERO 80))))))",
  "PERMITIR", [{"accion":"PERMITIR","protocolo":"tcp","destino":"192.168.1.10","puerto_destino":[80,80]}],
  ["iptables -A INPUT -d 192.168.1.10 -p tcp --dport 80 -j ACCEPT"],
  ["access-list 101 permit tcp any host 192.168.1.10 eq 80"]),
 ("bloquea ssh",
  "(S (SV (VACC bloquea) (SN (SERV ssh))))",
  "BLOQUEAR", [{"accion":"BLOQUEAR","protocolo":"tcp","puerto_destino":[22,22]}],
  ["iptables -A INPUT -p tcp --dport 22 -j DROP"],
  ["access-list 101 deny tcp any any eq 22"]),
 ("permite http desde 172.16.0.0/16",
  "(S (SV (SV (VACC permite) (SN (SERV http))) (SP (PORIG desde) (SN (DIR 172.16.0.0/16)))))",
  "PERMITIR", [{"accion":"PERMITIR","protocolo":"tcp","origen":"172.16.0.0/16","puerto_destino":[80,80]}],
  ["iptables -A INPUT -s 172.16.0.0/16 -p tcp --dport 80 -j ACCEPT"],
  ["access-list 101 permit tcp 172.16.0.0 0.0.255.255 any eq 80"]),
 ("bloquea el acceso a 10.0.0.5",
  "(S (SV (SV (VACC bloquea) (SN (DET el) (NUC (NTRAF acceso)))) (SP (PDEST a) (SN (DIR 10.0.0.5)))))",
  "BLOQUEAR", [{"accion":"BLOQUEAR","protocolo":"any","destino":"10.0.0.5"}],
  ["iptables -A INPUT -d 10.0.0.5 -j DROP"],
  ["access-list 101 deny ip any host 10.0.0.5"]),
]

regs = []
for i, (orden, arbol, intencion, fls, ipt, acl) in enumerate(ORO, 1):
    formas = orden.split()
    toks = [Token(f, f, tipo(f)) for f in formas]
    a = desde_corchetes(arbol, toks)
    terms = [t.terminal for t in toks]
    for r in reglas(a):
        assert any(rhs == r[1] for rhs, _ in G[r[0]]), (orden, r)
    p_oro = math.prod(next(p for rhs, p in G[A] if rhs == b) for A, b in reglas(a))
    arbs = todos(tuple(terms))
    print(f"{orden!r}: {len(arbs)} árboles; P(oro)={p_oro:.4g}; P(mejor)={arbs[0][0]:.4g}; oro es el mejor: {abs(arbs[0][0]-p_oro)<1e-15}")
    reg = {"id": f"mini-{i:03d}", "orden": orden, "orden_ruidosa": None, "intencion": intencion,
           "tokens": formas, "etiquetas": preterminales(a), "arbol": arbol, "formas_logicas": fls,
           "iptables": ipt, "acl": acl, "ambigua": len(arbs) > 1, "fuente": "sintetico"}
    assert validar_registro(reg) == [], validar_registro(reg)
    regs.append(reg)

with open("tests/fixtures/mini_corpus.jsonl", "w", encoding="utf-8") as f:
    for r in regs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("escrito", len(regs))

esperado = []
for orden, *_ in ORO:
    terms = [Token(f, f, tipo(f)).terminal for f in orden.split()]
    arbs = todos(tuple(terms))
    esperado.append({"orden": orden, "terminales": terms, "n_arboles": len(arbs),
                     "probs_ordenadas": [float(f"{p:.10g}") for p, _ in arbs]})
with open("tests/fixtures/mini_cyk_esperado.json", "w", encoding="utf-8") as f:
    json.dump({"_nota": "Resultados esperados del CYK con tests/fixtures/mini.pcfg, calculados por enumeración exhaustiva. Las probabilidades no están en logaritmo.",
               "casos": esperado}, f, ensure_ascii=False, indent=1)
print(json.dumps(esperado, ensure_ascii=False)[:600])
