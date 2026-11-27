# Proyecto · División del trabajo en 7 roles paralelos

**Proyecto:** traductor probabilístico de órdenes en español a reglas de firewall (iptables y ACL de Cisco)
**Capítulo:** 24 · *Natural Language Processing*, Russell & Norvig, AIMA 4.ª ed. global
**Entrega:** lunes 7 de diciembre de 2026 · **Sustentación:** 8–12 de diciembre (sin computador, papel y lápiz)
**Plan general:** [PLAN.md](PLAN.md)

> **Cómo se logra el paralelismo.** Todos programan a la vez contra los **contratos de la §2**, que se
> congelan el martes 24. Mientras un módulo ajeno no esté listo, cada quien usa los *stubs* y los
> *fixtures* de `tests/fixtures/`. Si alguien necesita cambiar un contrato, lo propone en el chat del
> grupo y lo aprueban el rol 1 y el dueño del módulo afectado. Nadie lo cambia por su cuenta.

---

## 0. Asignación de roles

| # | Rol | Módulos | Integrante | Revisor de código |
|---|---|---|---|---|
| 1 | Arquitectura sintáctica: PCFG y CYK | M5, M6, `arbol.py` | **Sergio Nicolas Siabatto Cleves** | Rol 2 (Sebastián Alarcón) |
| 2 | Semántica y generación de reglas | M7, M8, `objetos.py` | **Jose Sebastian Alarcon Bravo** | Rol 1 (Sergio Siabatto) |
| 3 | Modelos de lenguaje y corrector | M2 (n-gramas, suavizado, corrector) | **Juan Sebastián Umaña Camacho** | Rol 4 (Mateo Herrera) |
| 4 | Clasificación y etiquetado | M3 (Naive Bayes, regresión logística), M4 (HMM, Viterbi) | **Mateo Herrera Muñoz** | Rol 3 (Juan Sebastián Umaña) |
| 5 | Datos, tokenización y pruebas | M1, generador de corpus, conjunto humano, pruebas de integración | **Alejandro Argüello Muñoz** | Rol 7 (Angel Olivera) |
| 6 | Integración, interfaz y diseño | `pipeline.py`, CLI, GUI, diagramas UML, manuales | **Edwin Felipe Pinilla Peralta** | Rol 5 (Alejandro Argüello) |
| 7 | Experimentos, informe y sustentación | E1–E5, figuras, edición del PDF, correo, guía de sustentación | **Angel Humberto Olivera Pinzón** | Rol 6 (Edwin Pinilla) |

Asignación por **sorteo** el 23-nov-2026 a las 19:18 (Python: `random.Random(1939311477).shuffle` sobre la lista
de integrantes en el orden del documento del Taller 01; la semilla se generó con `secrets.randbits(32)` y
permite reproducir el resultado). El **revisor** lee cada cambio de su compañero antes de que entre a
`main` y es su suplente si se enferma o se atrasa.

### Matriz de responsabilidades (R = responsable, A = apoya)

| Entregable | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Contratos (§2) y esqueleto | R | A | A | A | A | A | |
| Gramática v1 y léxico | R | A | | | A | | |
| Tabla de servicios y objetos de red | | R | | | A | | |
| Corpus sintético y *treebank* | A | A | | | R | | |
| Conjunto humano (órdenes a mano) | A | A | A | A | R | A | A |
| Marco teórico | A | A | A | A | | | R (edición) |
| Descripción y justificación del problema | | A | | | R | | A |
| Diseño UML | A | A | | | | R | |
| Manual de usuario | | | | | | R | A |
| Manual técnico | A | A | A | A | A | R | |
| Escenarios E1–E5 | A (E3, E5) | | A (E2) | A (E4) | A | | R |
| PDF final, `.zip` y correo | | | | | | A | R |
| Guía de sustentación | A | A | A | A | A | A | R |

---

## 1. Reglas comunes

1. **Python 3.13**, solo stdlib + **NumPy** (estructuras de datos) + **matplotlib** (solo figuras de
   experimentos). Prohibido cualquier otro paquete.
2. **De NumPy solo arreglos y aritmética**: `np.zeros`, indexación, `np.log`, `np.exp`, `argmax`,
   sumas, productos. Nada que resuelva una parte del problema.
3. Todo algoritmo del capítulo se escribe desde cero, con un *docstring* que cite la sección y la página del
   libro (p. ej. `"""CYK probabilístico, AIMA §24.3, fig. 24.5, p. 888."""`).
4. Identificadores en español, sin tildes. Formato `black` opcional; lo importante es que sea legible.
5. **Probabilidades en logaritmo** (`math.log`) en todo el sistema para evitar *underflow*.
   Se usa `-math.inf` para probabilidad 0.
6. Aleatoriedad siempre con `random.Random(semilla)` o `np.random.default_rng(semilla)`, nunca global.
7. Cada módulo trae `tests/test_<modulo>.py` con `unittest`. Nada entra a `main` sin pruebas que pasen.
8. **Repositorio Git compartido** (GitHub privado sugerido). Una rama por rol (`rol1-cyk`, …),
   *pull request* a `main` revisado por el revisor de la §0.
9. **Seguimiento diario por el chat del grupo** (3 líneas antes de las 9 p. m.): qué hice, qué haré,
   qué me bloquea.
10. Reuniones cortas (30 min, virtuales) solo en los hitos H0–H5 (§4).

---

## 2. Contratos entre módulos (se congelan el martes 24)

El rol 1 los sube el martes 24 como esqueleto en `src/ordenred/`, con *stubs* que lanzan
`NotImplementedError`.

### C1 · Token (`tokenizador.py`, dueño: rol 5)

```python
@dataclass(frozen=True)
class Token:
    texto: str        # tal como lo escribió el usuario: "10.0.0.0/8"
    forma: str        # normalizado: minúsculas, sin tildes: "10.0.0.0/8"
    tipo: str         # "PALABRA" | "IP" | "CIDR" | "NUM" | "RANGO" | "IFAZ"
    @property
    def terminal(self) -> str:
        """Símbolo que ve la gramática: la forma si es PALABRA, si no "<IP>", "<CIDR>", "<NUM>", "<RANGO>", "<IFAZ>"."""

def tokenizar(texto: str) -> list[Token]
```

- `RANGO` = `1000-2000` o `1000:2000`; `IFAZ` = `eth0`, `wlan1`, `enp3s0`, `Gi0/1`.
- Contracciones del español: «al» → `a el`, «del» → `de el`, para que la gramática no las duplique.

### C2 · Formato de la gramática (`gramatica/reglas.pcfg`, dueño: rol 1)

```text
# comentario
S       -> SV                 [1.0]
SV      -> VACC SN            [0.55]
SV      -> SV SP              [0.45]
VACC    -> "bloquea"          [0.30]
DIR     -> "<CIDR>"           [0.40]
```

- Solo se admiten reglas **binarias** (`A -> B C`), **léxicas** (`A -> "w"`) y **unarias** (`A -> B`).
  El CYK aplica la clausura unaria.
- Las probabilidades del archivo son **iniciales**. M6 las reestima por conteo sobre el *treebank*.
- Las categorías preterminales (las que tienen reglas léxicas) son el **conjunto de etiquetas del HMM** (C6).

### C3 · Árbol (`arbol.py`, dueño: rol 1)

```python
@dataclass
class Arbol:
    etiqueta: str                    # "SV", "VACC", ...
    hijos: list["Arbol"]             # vacía en las hojas
    token: Token | None = None       # solo en preterminales
    logprob: float = 0.0             # log P del subárbol

def a_corchetes(a: Arbol) -> str     # "(S (SV (VACC bloquea) (SN ...)))"
def desde_corchetes(s: str, tokens: list[Token] | None = None) -> Arbol   # con tokens, las hojas quedan con su tipo real
def preterminales(a: Arbol) -> list[str]
def hojas(a: Arbol) -> list[Token]
def reglas(a: Arbol) -> list[tuple[str, tuple[str, ...]]]   # notación de reglas.pcfg; la usa M6 para contar
```

**Ya implementado en el esqueleto** (`src/ordenred/arbol.py`, con pruebas).

### C4 · Forma lógica (`semantica.py`, dueño: rol 2)

```python
@dataclass
class FormaLogica:
    accion: str                      # "PERMITIR" | "BLOQUEAR" | "RECHAZAR" | "LIMITAR" | "REDIRIGIR" | "ELIMINAR" | "LISTAR"
    protocolo: str = "any"           # "tcp" | "udp" | "icmp" | "any"
    origen: str | None = None        # IP o CIDR ya resuelto
    destino: str | None = None
    puerto_destino: tuple[int, int] | None = None    # (22, 22) o (1000, 2000)
    puerto_origen: tuple[int, int] | None = None
    cadena: str = "INPUT"            # "INPUT" | "OUTPUT" | "FORWARD"
    interfaz: str | None = None
    tasa: str | None = None          # "10/second" (LIMITAR)
    redirigir_a: str | None = None   # "192.168.1.10:8080" (REDIRIGIR)
    def a_logica(self) -> str:       # "Bloquear(x) ∧ Origen(x, 10.0.0.0/8) ∧ Puerto(x, 22) ∧ Proto(x, tcp)"
    def a_dict(self) -> dict         # para JSON
```

Una orden con coordinación («permite http y ssh») produce **una lista** de `FormaLogica`.

### C5 · Registro del corpus (`datos/*.jsonl`, dueño: rol 5)

Una línea JSON por orden:

```json
{"id": "sint-000123",
 "orden": "bloquea el ssh desde la red de invitados",
 "orden_ruidosa": "bloqea el ssh desde la red de invitdos",
 "intencion": "BLOQUEAR",
 "tokens": ["bloquea", "el", "ssh", "desde", "la", "red", "de", "invitados"],
 "etiquetas": ["VACC", "DET", "SERV", "PORIG", "DET", "NRED", "PREP", "NOMBRE"],
 "arbol": "(S (SV ...))",
 "formas_logicas": [{"accion": "BLOQUEAR", "protocolo": "tcp", "origen": "192.168.50.0/24", "puerto_destino": [22, 22]}],
 "iptables": ["iptables -A INPUT -s 192.168.50.0/24 -p tcp --dport 22 -j DROP"],
 "acl": ["access-list 101 deny tcp 192.168.50.0 0.0.0.255 any eq 22"],
 "ambigua": false, "fuente": "sintetico"}
```

Divisiones fijas con semilla 2026: `train.jsonl` (70 %), `dev.jsonl` (15 %), `test.jsonl` (15 %),
más `humano.jsonl` (se usa **solo para prueba**).

`src/ordenred/corpus.py` (ya implementado) trae `cargar_jsonl`, `guardar_jsonl` y `validar_registro`,
que verifica todo C5. En registros humanos fuera de la cobertura de la gramática, `arbol` y
`etiquetas` pueden ser `null`. `ambigua` significa «la gramática le da más de un árbol».

### C6 · Interfaces de los modelos

```python
# ngramas.py (rol 3)
class ModeloNGramas:
    def __init__(self, n: int, suavizado: str = "laplace", unidad: str = "palabra"):  # suavizado: "laplace" | "backoff" | "interpolacion"; unidad: "palabra" | "caracter"
    def entrenar(self, secuencias: list[list[str]]) -> None
    def ajustar_lambdas(self, dev: list[list[str]]) -> None     # interpolación
    def logprob(self, secuencia: list[str]) -> float
    def perplejidad(self, secuencias: list[list[str]]) -> float

# corrector.py (rol 3)
class Corrector:
    def __init__(self, vocabulario: set[str], lm_palabras: ModeloNGramas, lm_caracteres: ModeloNGramas)
    def corregir(self, tokens: list[Token]) -> tuple[list[Token], list[tuple[str, str, float]]]   # (corregidos, [(antes, despues, logprob)])

# clasificadores.py (rol 4)
class Clasificador(Protocol):
    def entrenar(self, X: list[list[str]], y: list[str]) -> None
    def predecir(self, tokens: list[str]) -> tuple[str, dict[str, float]]    # (clase, P(clase | tokens))
class NaiveBayes(Clasificador): ...
class RegresionLogistica(Clasificador): ...
    # guardar/cargar (agregado en el PR del rol 4) en NaiveBayes, RegresionLogistica y HMM:
    def guardar(self, ruta: str) -> None                         # JSON con "tipo" y "version"
    @classmethod
    def cargar(cls, ruta: str) -> "NaiveBayes"                   # idem RegresionLogistica / HMM

# hmm.py (rol 4)
class HMM:
    def entrenar(self, oraciones: list[list[tuple[str, str]]]) -> None    # [(palabra, etiqueta)]
    def viterbi(self, palabras: list[str]) -> tuple[list[str], float]
    # agregado en el PR del rol 4: lo que el pipeline le pasa al CYK como etiquetas_unk
    def etiquetas_unk(self, palabras: list[str], posiciones: list[int],
                      permitidas: set[str] | None = None) -> dict[int, str]

# pcfg.py y cyk.py (rol 1)
class PCFG:
    @classmethod
    def desde_archivo(cls, ruta: str) -> "PCFG"
    def estimar_desde_treebank(self, arboles: list[Arbol], suavizado: float = 0.0) -> None
    def guardar(self, ruta: str) -> None
class ParserCYK:
    def __init__(self, pcfg: PCFG)
    def analizar(self, terminales: list[str], k: int = 1, etiquetas_unk: dict[int, str] | None = None,
                 tokens: list[Token] | None = None) -> list[Arbol]   # k mejores, ordenados
    # tokens (agregado en el PR #1): si se pasan, las hojas llevan el token real (IP, puerto…), que es
    # lo que necesitan el intérprete (rol 2) y la comparación con el árbol de oro (E1). El pipeline debe pasarlos.

# objetos.py (rol 2)
class TablaObjetos:
    @classmethod
    def desde_archivo(cls, ruta: str) -> "TablaObjetos"
    def servicio(self, nombre: str) -> tuple[str, int | None] | None   # "ssh" -> ("tcp", 22); "ping" -> ("icmp", None)
    def red(self, nombre: str) -> str | None                           # "invitados" -> "192.168.50.0/24"

# semantica.py (rol 2)
class Interprete:
    def __init__(self, objetos: "TablaObjetos")
    def interpretar(self, arbol: Arbol) -> list[FormaLogica]

# generador_reglas.py (rol 2)
def a_iptables(fls: list[FormaLogica]) -> list[str]
def a_acl_cisco(fls: list[FormaLogica], numero: int = 101) -> list[str]
```

**Acuerdos de los modelos del rol 4 (agregados en el PR del rol 4; compatibles con lo anterior):**

- **Entrada.** `Clasificador.entrenar/predecir` y `HMM.entrenar/viterbi` reciben los **terminales**
  (`Token.terminal`: `"<CIDR>"`, `"<IP>"`, `"<NUM>"`…), no la forma escrita (`"10.0.0.0/8"`). Así una IP nueva
  no es una palabra desconocida y los modelos generalizan. En el corpus (C5) `tokens` trae formas: quien
  entrena convierte con `tokenizar`.
- **Persistencia.** `guardar(ruta)` / `cargar(ruta)` en JSON (`"tipo"`, `"version"`). Naive Bayes y el HMM
  guardan conteos enteros y recalculan los logaritmos al cargar (ida y vuelta exacta); la regresión logística
  guarda sus pesos. Los modelos entrenados van en `modelos/` (no se versiona); lo usa `cli entrenar` (rol 6).
- **Palabras `<UNK>`.** El pipeline (rol 6) decide qué posiciones están fuera del léxico de la PCFG
  (`PCFG.indice_lexico`). El HMM etiqueta la **oración completa** (el contexto de las palabras conocidas es lo
  que permite acertar) y `HMM.etiquetas_unk` devuelve solo esas posiciones, en el formato de
  `ParserCYK.analizar(..., etiquetas_unk=...)`. `permitidas` restringe la etiqueta a preterminales que existen
  en la gramática, para que el CYK nunca reciba una categoría sin reglas.
- **Nota para el rol 1.** Hoy el CYK le asigna log P = 0 a la etiqueta del HMM en esa posición, así que solo
  cuenta cuál etiqueta se eligió, no con qué confianza. No bloquea; se deja anotado para E3/E4.

### C7 · Resultado del pipeline (`pipeline.py`, dueño: rol 6)

```python
@dataclass
class Resultado:
    texto: str
    tokens: list[Token]
    correcciones: list[tuple[str, str, float]]
    intencion: str
    prob_intencion: dict[str, float]
    etiquetas_unk: dict[int, str]          # posición -> etiqueta propuesta por el HMM
    arboles: list[Arbol]                   # k mejores
    formas_logicas: list[FormaLogica]      # del mejor árbol
    iptables: list[str]
    acl: list[str]
    advertencias: list[str]                # "orden ambigua", "intención y árbol no coinciden", ...
    tiempos_ms: dict[str, float]           # por etapa (para E5)

class Pipeline:
    def __init__(self, config: dict)       # qué suavizado, qué clasificador, k, usar corrector sí/no, ...
    def traducir(self, texto: str) -> Resultado
```

La CLI, la GUI y los experimentos **solo** usan `Pipeline` y `Resultado`.

### C8 · Estructura del repositorio y dueños de archivos

```
Proyecto/
├── PLAN.md  DIVISION_DE_TRABAJO.md  README.md (6)  requirements.txt (6)
├── src/ordenred/
│   ├── tokenizador.py (5)   ngramas.py (3)   corrector.py (3)
│   ├── clasificadores.py (4)   hmm.py (4)
│   ├── arbol.py (1)   pcfg.py (1)   cyk.py (1)
│   ├── semantica.py (2)   objetos.py (2)   generador_reglas.py (2)
│   ├── pipeline.py (6)   cli.py (6)   gui.py (6)   corpus.py (5)
│   └── gramatica/ reglas.pcfg (1)  objetos.json (2)
├── datos/ generador_corpus.py (5)  ruido.py (5)  train|dev|test|humano.jsonl (5)
├── experimentos/ runner.py (7)  e1..e5_*.py (7 + apoyos)  resultados/*.csv  figuras/*.pdf
├── tests/ fixtures/ (1 y 5)  test_*.py (cada dueño)  test_contratos.py (1)  test_integracion.py (5)
├── diseno/ *.puml o *.drawio + PNG exportados (6)
└── informe/ main.tex (7)  secciones/*.tex (cada autor)  figuras/
```

### Boceto de la gramática (lo cierran los roles 1, 2 y 5 el jueves 26)

| Categoría | Ejemplos |
|---|---|
| `VACC` acción | bloquea, bloquear, deniega, rechaza, permite, acepta, habilita, limita, redirige, elimina, borra, muestra, lista |
| `NTRAF` tráfico | tráfico, acceso, conexiones, paquetes, peticiones |
| `SERV` servicio | ssh, http, https, dns, ftp, smtp, rdp, telnet, ping |
| `PROTO` protocolo | tcp, udp, icmp |
| `PORIG` / `PDEST` / `PPUERTO` / `PIFAZ` | desde, de / hacia, a, para / por, en / en, por |
| `NPUERTO` | puerto, puertos |
| `NRED` / `NOMBRE` | red, servidor, equipo, host / invitados, web, dmz, administración |
| `DET`, `CONJ`, `PREP` | el, la, los, las, todo / y, o / de |
| Especiales | `<IP>`, `<CIDR>`, `<NUM>`, `<RANGO>`, `<IFAZ>` |

Ambigüedades **buscadas a propósito** para E3:
- **Coordinación:** «permite http y ssh desde la red de invitados», donde «desde…» puede afectar a ambos servicios o solo a ssh.
- **Adjunción del sintagma preposicional:** «bloquea el acceso al servidor de la red de invitados»: ¿el servidor *de* la red de invitados (destino) o el acceso *desde* ella?
- **Puerto de origen o de destino:** «rechaza las conexiones de 10.0.0.5 por el puerto 22».

---

## 3. Tareas por rol

Formato de cada rol: **misión → depende de / entrega a → tareas fechadas → terminado cuando → informe → sustentación.**

### Rol 1 · Arquitectura sintáctica: PCFG y CYK (M5, M6)

**Misión:** dueño de la gramática y del analizador sintáctico, que es el corazón probabilístico del sistema.

- **Depende de:** rol 5 (*treebank* para M6, desde el jueves 26); usa *fixtures* mientras tanto.
- **Entrega a:** roles 2, 4 y 6 (`Arbol`, `ParserCYK`), rol 5 (gramática para el generador), rol 7 (E3, E5).

**Tareas**
- [x] **Lun 23:** esqueleto `src/ordenred/` con todos los contratos de la §2 como *stubs*; `tests/fixtures/` con `mini.pcfg` (29 reglas), `mini_corpus.jsonl` (5 órdenes) y `mini_cyk_esperado.json`. *(Hecho.)*
- [x] **Lun 23:** `arbol.py` completo (`a_corchetes`, `desde_corchetes`, `preterminales`, `hojas`, `reglas`) con pruebas. *(Hecho.)*
- [ ] **Mar 24:** revisar el esqueleto con el grupo en H0 y subirlo al repositorio compartido.
- [ ] **Mié 25–Jue 26:** `gramatica/reglas.pcfg` v1 junto con los roles 2 y 5: unas 60–120 reglas y un léxico de ≈ 150 palabras, que incluya las ambigüedades de la §2.
- [ ] **Mié 25–Vie 27:** `pcfg.py`: lectura y validación del archivo (que las reglas de cada no terminal sumen 1 y que estén en el formato permitido), `guardar`.
- [x] **Jue 26–Sáb 28:** `cyk.py`: CYK probabilístico (AIMA fig. 24.5) en log-probabilidades y punteros de retroceso; clausura unaria; reconstrucción del árbol. *(PR #1. Se implementó con diccionarios, no con tabla NumPy: decidir si se cambia o si NumPy se justifica con otros módulos.)*
- [x] **Sáb 28–Dom 29:** *k* mejores árboles (*k*-best por celda, *k* ≤ 5); uso de `etiquetas_unk` del HMM para posiciones `<UNK>`.
- [ ] **Dom 29–Lun 30:** M6, estimación de máxima verosimilitud de la PCFG por conteo sobre `train.jsonl` (AIMA §24.3.2), con suavizado aditivo opcional; comparación de la verosimilitud en `dev`.
- [ ] **Lun 30–Mar 1:** apoyo al rol 7 con el código específico de **E3** (PCFG aprendida vs. uniforme; rango de la interpretación correcta) y **E5** (tiempo vs. longitud y tamaño de la gramática).
- [ ] **Mié 2:** corrección de errores de integración; congelamiento.

**Terminado cuando:** el CYK reproduce a mano la tabla de una orden de 4 palabras con la mini gramática;
la exactitud de árbol en `dev` es ≥ 95 % para órdenes limpias; *k*-best devuelve árboles distintos y
ordenados; las pruebas pasan.

**Informe:** marco teórico de gramáticas libres de contexto probabilísticas, FNC, CYK y aprendizaje de
la PCFG (≈ 3 págs.); en el manual técnico, `pcfg.py`/`cyk.py` y el formato de la gramática (≈ 1,5 págs.);
pseudocódigo de CYK para el diseño.

**Sustentación:** ejercicio de tabla CYK con una gramática de 6–8 reglas y una orden de 4 palabras;
ejercicio de dos árboles de una orden ambigua y cuál gana. Prepara también la versión resuelta.

---

### Rol 2 · Semántica y generación de reglas (M7, M8)

**Misión:** convertir el árbol en significado (gramática aumentada) y el significado en reglas reales.

- **Depende de:** rol 1 (`Arbol`; usa los árboles de los *fixtures* desde el martes 24).
- **Entrega a:** rol 5 (el generador de corpus usa `a_iptables`/`a_acl_cisco` para las etiquetas de oro), rol 6.

**Tareas**
- [ ] **Mar 24–Mié 25:** `gramatica/objetos.json` y `objetos.py` (`TablaObjetos`): servicios → (protocolo, puerto) (ssh → tcp/22, dns → udp/53, ping → icmp, …) y redes con nombre (invitados → `192.168.50.0/24`, dmz, servidor web, administración…). Validación de IP/CIDR **a mano** (sin `ipaddress`, o declararlo como stdlib y usar solo su validación).
- [ ] **Mié 25–Jue 26:** junto con roles 1 y 5, definir la **función semántica de cada regla** de la gramática (tabla regla → composición), que es la parte de «gramática aumentada» (AIMA §24.4).
- [ ] **Jue 26–Sáb 28:** `semantica.py`: `FormaLogica`, `a_logica()`, `Interprete.interpretar` por composición recursiva sobre el árbol; manejo de coordinación (una orden → varias formas lógicas), valores por omisión y detección de contradicciones (p. ej. «tcp» con «ping»).
- [ ] **Vie 27–Dom 29:** `generador_reglas.py`: `a_iptables` (filtro `-A INPUT/OUTPUT/FORWARD`, `-s`, `-d`, `-p`, `--dport`, `--sport`, `-i`, `-m limit --limit`, DNAT para REDIRIGIR, `-D` para ELIMINAR, `-L` para LISTAR) y `a_acl_cisco` (ACL extendida numerada, conversión de CIDR a máscara *wildcard*, `eq`/`range`, `permit`/`deny`).
- [ ] **Lun 30–Mar 1:** pruebas con todos los árboles de `dev`; tabla de casos borde (sin origen, rango de puertos, ICMP sin puerto).
- [ ] **Mié 2:** integración y congelamiento.

**Terminado cuando:** las formas lógicas de los árboles de oro de `dev` coinciden al 100 % con las de oro
del generador; las reglas generadas tienen sintaxis válida (lista de verificación escrita con ejemplos de
`man iptables` y de la documentación de Cisco IOS).

**Informe:** marco teórico de gramáticas aumentadas, semántica composicional y forma lógica (≈ 2 págs.),
más un apartado corto sobre iptables y ACL (≈ 1 pág.); en el manual técnico, `semantica.py`, `objetos.json`
y `generador_reglas.py`.

**Sustentación:** construir a mano la forma lógica desde un árbol y escribir la regla iptables y la ACL;
convertir un CIDR a *wildcard*.

---

### Rol 3 · Modelos de lenguaje y corrector (M2)

**Misión:** que las órdenes con errores de digitación lleguen corregidas al analizador.

- **Depende de:** rol 5 (`tokenizar`, corpus). Mientras tanto usa una lista de 50 órdenes de los *fixtures*.
- **Entrega a:** rol 6 (`Corrector` en el pipeline), rol 7 (E2).

**Tareas**
- [ ] **Mar 24–Jue 26:** `ngramas.py`: conteos de n-gramas (n = 1…3) para palabras y para caracteres, con marcas de inicio y fin; `<UNK>` por umbral de frecuencia (AIMA §24.1.4).
- [ ] **Jue 26–Vie 27:** tres suavizados: **Laplace** (sumar 1), **backoff** (estilo Katz simplificado o *stupid backoff*, documentado) e **interpolación lineal** con λ ajustados en `dev` por búsqueda en rejilla (λ₃ + λ₂ + λ₁ = 1); `perplejidad`.
- [ ] **Vie 27–Dom 29:** `corrector.py` de canal ruidoso: candidatos a distancia de edición ≤ 2 (Damerau-Levenshtein implementada a mano) dentro del vocabulario, P(error | palabra) por tipo de edición y P(palabra | contexto) por el modelo de palabras. Para palabras sin candidatos se usa el modelo de caracteres. No se corrigen tokens IP/CIDR/NUM.
- [ ] **Lun 30–Mar 1:** código de **E2**: exactitud con y sin corrector según la tasa de ruido; perplejidad de los tres suavizados en `test` y `humano`.
- [ ] **Mié 2:** integración y congelamiento.

**Terminado cuando:** recupera ≥ 90 % de las palabras alteradas con 1 edición en `dev`; no «corrige» palabras
válidas en más del 2 % de los casos; perplejidad con interpolación < Laplace en `dev`.

**Informe:** marco teórico de modelos de lenguaje, n-gramas como cadenas de Markov, suavizado y canal
ruidoso (≈ 3 págs.; es la parte más ligada a la asignatura); en el manual técnico, `ngramas.py` y `corrector.py`.

**Sustentación:** bigramas a mano sobre 5 órdenes; probabilidad de una orden con Laplace y con interpolación
(λ dados); elegir la corrección de «bloqea» entre 2–3 candidatos con una tabla dada.

---

### Rol 4 · Clasificación y etiquetado (M3, M4)

**Misión:** decir qué quiere el usuario (intención) y asignar categoría gramatical a las palabras desconocidas.

- **Depende de:** rol 5 (corpus con `intencion` y `etiquetas`), rol 1 (conjunto de etiquetas = preterminales).
- **Entrega a:** rol 6, rol 1 (`etiquetas_unk` para el CYK), rol 7 (E4).

**Tareas**
- [ ] **Mar 24–Jue 26:** `NaiveBayes` multinomial con bolsa de palabras y suavizado de Laplace, en log-probabilidades (AIMA §24.1.1).
- [ ] **Jue 26–Sáb 28:** `RegresionLogistica` multiclase (*softmax*) con descenso por gradiente en NumPy, regularización L2 y vectorizador propio (vocabulario → índice); semilla fija (AIMA §24.1.6, discriminativo frente a generativo).
- [ ] **Vie 27–Dom 29:** `hmm.py`: estimación de transiciones y emisiones por conteo con suavizado; manejo de palabras desconocidas por sufijo o forma; **Viterbi** en log con tabla NumPy y retroceso (AIMA §24.1.6).
- [ ] **Dom 29–Lun 30:** contrato con el rol 1: para cada posición `<UNK>`, el HMM propone la etiqueta y el CYK la usa como preterminal.
- [ ] **Lun 30–Mar 1:** código de **E4**: curvas de aprendizaje (10 %, 25 %, 50 %, 100 % de `train`) de NB vs. RL; exactitud del HMM en palabras desconocidas; matriz de confusión de intenciones.
- [ ] **Mié 2:** integración y congelamiento.

**Terminado cuando:** intención ≥ 95 % en `dev` limpio; Viterbi coincide con la búsqueda exhaustiva en
oraciones cortas (prueba); exactitud de etiquetado ≥ 90 % en `dev`.

**Informe:** marco teórico de Naive Bayes, regresión logística, modelos generativos y discriminativos,
HMM y Viterbi (≈ 3 págs.); en el manual técnico, `clasificadores.py` y `hmm.py`.

**Sustentación:** clasificar una orden con una tabla de conteos (3 clases, Laplace); llenar una tabla de
Viterbi de 3 palabras × 3 etiquetas.

---

### Rol 5 · Datos, tokenización y pruebas (M1, corpus)

**Misión:** que todos tengan datos de calidad y que el sistema completo esté probado.

- **Depende de:** gramática v1 (rol 1, jueves 26), `objetos.json` y `a_iptables`/`a_acl_cisco` (rol 2).
- **Entrega a:** todos (corpus), rol 3 (`tokenizar`).

**Tareas**
- [ ] **Lun 23–Mar 24 (urgente):** pedir a **los 7 integrantes 25 órdenes cada uno escritas a mano, sin ver la gramática** (≈ 175 órdenes realistas). Recogerlas antes del martes en la noche: después de ver la gramática ya no sirven. Plantilla con las 7 intenciones.
- [ ] **Mar 24–Mié 25:** `tokenizador.py`: expresiones regulares para IP, CIDR, NUM, RANGO, IFAZ; normalización (minúsculas, tildes); separación de contracciones; pruebas con casos borde (`10.0.0.256`, `/33`, `eth0.100`).
- [ ] **Mié 25–Jue 26:** aportar a la gramática v1 los fenómenos vistos en las órdenes humanas.
- [ ] **Jue 26–Sáb 28:** `generador_corpus.py`: muestrear primero una forma lógica, luego realizarla en texto **construyendo al mismo tiempo el árbol de oro** (así la forma lógica de oro no depende del intérprete del rol 2); etiquetas, intención, reglas de oro; ≈ 5000 órdenes; marcar `ambigua`.
- [ ] **Vie 27–Sáb 28:** `ruido.py`: sustitución por teclas vecinas (teclado QWERTY español), omisión, inserción, transposición, con tasa configurable y semilla.
- [ ] **Sáb 28–Lun 30:** anotar `humano.jsonl` (forma lógica, intención y reglas de oro; el árbol solo si la orden queda dentro de la gramática); marcar las que quedan fuera de cobertura.
- [ ] **Dom 29–Mar 1:** `tests/test_integracion.py`: 30 órdenes de punta a punta (texto → reglas) contra el pipeline.
- [ ] **Mié 2:** corpus final congelado (`v1.0`), con estadísticas (tamaño, distribución de intenciones, longitud).

**Terminado cuando:** corpus reproducible con un comando y una semilla; validación automática de que cada
registro cumple C5; las pruebas de integración corren con `python -m unittest`.

**Informe:** **descripción y justificación del problema** completa (≈ 3 págs.: contexto, errores de
configuración de firewalls, por qué lenguaje natural, alcance y límites); descripción de los datos (≈ 1 pág.);
en el manual técnico, tokenizador y generador.

**Sustentación:** tokenizar una orden con IP, CIDR y puerto, y decir qué va a `<UNK>`; explicar cómo se
generó el corpus y por qué no es circular.

---

### Rol 6 · Integración, interfaz y diseño (pipeline, CLI, GUI, UML, manuales)

**Misión:** unir todo, hacerlo usable y documentar la arquitectura.

- **Depende de:** todos (contratos desde el martes 24; implementaciones reales a medida que llegan).
- **Entrega a:** rol 7 (pipeline para los experimentos), profesor (manuales).

**Tareas**
- [ ] **Mar 24–Mié 25:** `pipeline.py` con `Resultado`, cableado con los *stubs* (que el flujo exista desde el día 1); `config` por omisión; medición de `tiempos_ms` por etapa.
- [ ] **Mié 25–Vie 27:** **diseño UML** en `diseno/` (PlantUML o draw.io): casos de uso, componentes/bloques del pipeline, clases (a partir de los contratos), secuencia de `traducir`, actividad del flujo con advertencias. Exportar a PNG para el informe.
- [ ] **Jue 26–Sáb 28:** `cli.py` (`argparse`): `traducir "orden"`, `--formato iptables|acl|ambos`, `--k`, `--sin-corrector`, `--clasificador nb|rl`, `--arbol` (árbol en ASCII), `entrenar` (reentrena y guarda modelos en `modelos/`), `lote archivo.txt`.
- [ ] **Sáb 28–Dom 29 (H2):** integración del núcleo: orden limpia → regla con M1, M5, M6, M7 y M8 reales.
- [ ] **Lun 30–Jue 3:** `gui.py` (tkinter): campo de orden, botón «Traducir», correcciones resaltadas, barras de P(intención), árbol dibujado en `Canvas` con probabilidades, pestañas con los *k* mejores árboles, formas lógicas, iptables y ACL con botón «Copiar», panel de advertencias.
- [ ] **Mié 2 (H3):** integración completa M1–M9; `README.md`; `requirements.txt` con versiones exactas.
- [ ] **Vie 4–Sáb 5:** manuales.

**Terminado cuando:** `python -m ordenred.cli traducir "..."` y `python -m ordenred.gui` funcionan en una
máquina limpia siguiendo **solo** el manual (lo prueba otro integrante); los diagramas coinciden con el código final.

**Informe:** **diseño de la aplicación** completo (≈ 5 págs. con diagramas); **manual de usuario**
(≈ 4 págs. con capturas: instalación, CLI, GUI, ejemplos, errores frecuentes); **manual técnico**
(estructura, contratos, cómo extender la gramática o agregar un servicio, cómo reentrenar; integra los
apartados de cada rol).

**Sustentación:** dibujar de memoria el diagrama de componentes y la secuencia de una traducción;
explicar qué pasa en cada etapa con una orden de ejemplo.

---

### Rol 7 · Experimentos, informe y sustentación

**Misión:** convertir el sistema en evidencia (escenarios, figuras, análisis) y en entregables listos.

- **Depende de:** pipeline (rol 6), código específico de cada escenario (roles 1, 3, 4), corpus (rol 5).
- **Entrega a:** profesor (PDF, `.zip`, correo); grupo (guía de sustentación).

**Tareas**
- [ ] **Mar 24–Mié 25:** `informe/main.tex`: plantilla con portada (UNAL, asignatura, profesor, integrantes), índice y una sección por literal (§5); cada autor escribe en `secciones/<x>.tex`.
- [ ] **Mié 25–Sáb 28:** `experimentos/runner.py`: correr un escenario con varias semillas (≥ 5), guardar CSV, calcular media e intervalo de confianza del 95 %; funciones de figuras con matplotlib en estilo único.
- [ ] **Sáb 28–Dom 29:** **protocolo de cada escenario** E1–E5 (§6): hipótesis, variables, métricas, configuración y figuras esperadas. Lo revisan los dueños de cada módulo.
- [ ] **Mar 1–Vie 4:** correr E1–E5 con el código congelado; figuras y tablas; **análisis de resultados** con los dueños de cada módulo.
- [ ] **Jue 3–Sáb 5:** integrar todas las secciones; coherencia de notación (misma notación de probabilidades que el libro); referencias; resumen ≤ 250 palabras.
- [ ] **Vie 4:** recoger los **datos completos** de los 7 (nombre como aparece en el SIA, documento, código, correo UNAL) y la frase del rol de cada uno.
- [ ] **Sáb 5–Dom 6:** guía de sustentación (§7), simulacros 1 y 2.
- [ ] **Lun 7:** `.zip` (PDF + código + datos + `requirements.txt`) y redacción del correo; lo envía un solo integrante.

**Terminado cuando:** todas las figuras se regeneran con un comando; cada escenario tiene hipótesis,
resultado y explicación; el PDF compila sin advertencias y cubre los 6 literales.

**Informe:** **escenarios de prueba y análisis** completo (≈ 8 págs.); conclusiones (≈ 1 pág.); edición general;
introducción del marco teórico (PLN y su relación con modelos estocásticos, ≈ 1 pág.).

**Sustentación:** explicar cada figura de E1–E5 y por qué sale así; coordina los simulacros.

---

## 4. Cronograma con hitos de sincronización

| Fecha | Hito | Condición para cumplirlo |
|---|---|---|
| **Mar 24** | **H0 · arranque** | Roles asignados; repositorio creado; contratos y esqueleto en `main`; las 175 órdenes humanas recogidas; `arbol.py` y `tokenizador.py` utilizables |
| **Jue 26** | **H1 · gramática y datos** | `reglas.pcfg` v1 y `objetos.json` congelados; tabla semántica por regla; corpus v0 (≈ 500) disponible; cada módulo con sus pruebas sobre *fixtures* |
| **Dom 29** | **H2 · núcleo integrado** | Orden limpia → regla por CLI (M1, M5, M6, M7, M8 reales); M2, M3 y M4 funcionando por separado con el corpus; diagramas UML v1; corpus v1 completo |
| **Mié 2** | **H3 · congelamiento de código** | M1–M9 integrados; pruebas de integración en verde; desde aquí **solo corrección de errores**; empiezan las corridas finales de E1–E5 |
| **Vie 4** | **H4 · resultados y borradores** | Figuras y tablas de E1–E5; GUI terminada; borrador de todas las secciones del informe; datos de los 7 integrantes recogidos |
| **Dom 6** | **H5 · cierre** | PDF final revisado por al menos dos personas distintas a su autor; `.zip` probado en máquina limpia; guía de sustentación; simulacro 2 |
| **Lun 7** | **Entrega** | Correo enviado (antes del mediodía, como margen) |
| 8–12 dic | Sustentación | Simulacro extra la víspera |

### Vista por días (quién hace qué al mismo tiempo)

| Día | Rol 1 | Rol 2 | Rol 3 | Rol 4 | Rol 5 | Rol 6 | Rol 7 |
|---|---|---|---|---|---|---|---|
| Lun 23 | Esqueleto | Objetos (inicio) | Lectura §24.1.2–4 | Lectura §24.1.1, 24.1.6 | Órdenes humanas | Lectura y bocetos UML | Plantilla LaTeX |
| Mar 24 | `arbol.py`, *fixtures* | `objetos.json` | n-gramas | Naive Bayes | Tokenizador | Pipeline con *stubs* | Plantilla LaTeX |
| Mié 25 | Gramática v1 | Gramática v1 (semántica) | n-gramas | Naive Bayes | Gramática v1, tokenizador | UML | Runner |
| Jue 26 | `pcfg.py`, CYK | Intérprete | Suavizados | Regresión logística | Generador de corpus | UML, CLI | Runner |
| Vie 27 | CYK | Intérprete, iptables | Suavizados, corrector | Regresión logística, HMM | Generador, ruido | CLI | Runner, figuras |
| Sáb 28 | CYK *k*-best | iptables, ACL | Corrector | HMM | Generador, ruido | CLI, integración | Protocolos E1–E5 |
| Dom 29 | *k*-best, M6 | ACL | Corrector | Viterbi | Anotar humano | **Integración H2** | Protocolos E1–E5 |
| Lun 30 | M6 | Casos borde | E2 | UNK→CYK, E4 | Anotar humano, integración | GUI | Pruebas de escenarios |
| Mar 1 | E3, E5 | Casos borde | E2 | E4 | Pruebas de integración | GUI | Corridas |
| Mié 2 | **Congelamiento H3** | | | | Corpus v1.0 | Integración completa | Corridas |
| Jue 3 | Informe | Informe | Informe | Informe | Informe | GUI, informe | Corridas, análisis |
| Vie 4 | Análisis E3/E5 | Informe | Análisis E2 | Análisis E4 | Informe | Manuales | Integración del informe |
| Sáb 5 | Ejercicio de sustentación | Ejercicio | Ejercicio | Ejercicio | Ejercicio | Manuales, prueba en máquina limpia | Guía, simulacro 1 |
| Dom 6 | Revisión cruzada | Revisión cruzada | Revisión cruzada | Revisión cruzada | Revisión cruzada | Revisión cruzada | **H5**, simulacro 2 |
| Lun 7 | | | | | | | Envío |

**Ruta crítica:** contratos (1) → gramática v1 (1, 2, 5) → generador de corpus (5) → M6 y entrenamiento de
modelos (1, 3, 4) → integración (6) → experimentos (7) → informe (7). Si la gramática v1 se atrasa, se
atrasa todo: **el jueves 26 es innegociable**.

---

## 5. Reparto del informe (un solo PDF)

| § | Sección | Autor principal | Aportan | Págs. aprox. |
|---|---|---|---|---|
| — | Portada, integrantes y roles, resumen (≤ 250 palabras) | 7 | todos | 2 |
| a | **Marco teórico** | 7 (introducción y edición) | 3: n-gramas y suavizado · 4: NB, RL, HMM, Viterbi · 1: PCFG, CYK, aprendizaje · 2: gramáticas aumentadas, semántica, iptables y ACL | 12 |
| b | **Descripción y justificación del problema** | 5 | 2, 7 | 3 |
| c | **Diseño de la aplicación** (UML, pseudocódigo) | 6 | 1 (pseudocódigo CYK), 4 (Viterbi), 3 (corrector) | 6 |
| d | **Código fuente** (descripción + referencia al `.zip`, versiones) | 6 | todos | 1 |
| e | **Manual de usuario** | 6 | 7 (capturas) | 4 |
| e | **Manual técnico** | 6 | cada dueño su módulo (≈ 1 pág. c/u) | 8 |
| f | **Escenarios de prueba y análisis** (E1–E5) | 7 | 1 (E3, E5), 3 (E2), 4 (E4), 5 (datos) | 8 |
| — | Conclusiones, trabajo futuro, referencias | 7 | todos | 2 |

**Revisión cruzada (domingo 6):** 1 revisa la sección a de 3 y 4; 3 revisa la de 1 y 2; 2 revisa b y c;
5 revisa e; 4 revisa f; 6 y 7 revisan el PDF completo.

---

## 6. Escenarios de prueba: responsables

| # | Escenario | Responsable del código | Corre y analiza | Métricas |
|---|---|---|---|---|
| E1 | Órdenes limpias dentro de la gramática (`test` y `humano` en cobertura) | 6 (pipeline) | 7 | Exactitud de intención, de árbol, de forma lógica y de regla exacta |
| E2 | Ruido 0, 5, 10, 15 y 20 % | 3 | 7 + 3 | Exactitud con y sin corrector; perplejidad de Laplace, backoff e interpolación |
| E3 | Ambigüedad (subconjunto `ambigua = true`) | 1 | 7 + 1 | PCFG aprendida vs. uniforme; rango de la interpretación correcta en los *k* mejores |
| E4 | Tamaño de entrenamiento y palabras desconocidas | 4 | 7 + 4 | Curvas de aprendizaje NB vs. RL; exactitud con y sin HMM para `<UNK>` |
| E5 | Escalabilidad del CYK | 1 | 7 + 1 | Tiempo vs. longitud n y vs. tamaño de gramática; ajuste a O(n³·\|G\|) |

El literal f exige mínimo 3. Si el tiempo no alcanza, E5 y E4 son los primeros que se recortan.

---

## 7. Sustentación (papel y lápiz)

Cada rol entrega el **sábado 5** su ejercicio para resolver a mano (enunciado + solución verificada con el
código, en `informe/sustentacion/`). El rol 7 los une en la **guía de sustentación**.

**Todos deben poder resolver todos los ejercicios**, no solo el de su módulo:

| Ejercicio | Prepara | Tiempo objetivo a mano |
|---|---|---|
| Tokenizar una orden y marcar `<UNK>` | 5 | 2 min |
| Bigramas + Laplace + interpolación | 3 | 8 min |
| Corrección por canal ruidoso | 3 | 5 min |
| Naive Bayes con tabla de conteos | 4 | 5 min |
| Tabla de Viterbi 3 × 3 | 4 | 10 min |
| Tabla CYK de 4 palabras | 1 | 12 min |
| Dos árboles de una orden ambigua: cuál gana | 1 | 5 min |
| Árbol → forma lógica → iptables y ACL | 2 | 6 min |
| Diagrama de componentes y secuencia de memoria | 6 | 5 min |
| Explicar una figura de resultados | 7 | 3 min |

- **Simulacro 1 (sábado 5):** cada integrante saca al azar un ejercicio **que no es suyo**.
- **Simulacro 2 (domingo 6):** dos ejercicios al azar por persona más una pregunta abierta del capítulo 24.
- **Víspera de la sustentación:** repaso de los ejercicios que más fallaron.

---

## 8. Riesgos de la ejecución en paralelo

| Riesgo | Mitigación | Responsable |
|---|---|---|
| Alguien cambia un contrato y rompe a los demás | Contratos congelados en H0; cambios solo con aprobación del rol 1 y del dueño | 1 |
| La gramática v1 se atrasa | Fecha fija el jueves 26; se puede congelar una v1 pequeña y ampliarla en v2 sin cambiar categorías | 1, 2, 5 |
| Integración tardía «big bang» | El pipeline existe con *stubs* desde el martes 24; integración parcial en H2 | 6 |
| Un integrante se atrasa o se enferma | Su revisor (§0) es su suplente; seguimiento diario por el chat | todos |
| Corpus circular (el generador y el intérprete comparten errores) | El generador parte de la forma lógica, no del intérprete; el conjunto humano es independiente | 5 |
| Experimentos sin tiempo al final | Runner listo desde el sábado 28; congelamiento del código el miércoles 2 | 7 |
| Nadie domina el sistema completo en la sustentación | Ejercicios cruzados y dos simulacros | 7 |
| Trámites con el profesor (IA, NumPy y matplotlib, grupo de 7) | Pendiente del grupo; no bloquea el desarrollo | grupo |
