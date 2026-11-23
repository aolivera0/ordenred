# Proyecto de curso — Plan (v1, 23-nov-2026)

**Asignatura:** Modelos Estocásticos y Simulación en Computación y Comunicaciones (UNAL) · Prof. Jorge Eduardo Ortiz Triviño
**Capítulo base:** 24 · *Natural Language Processing*, Russell & Norvig, AIMA 4.ª ed. global (pp. 874–906)
**Entrega:** lunes 7 de diciembre de 2026 (respuesta al correo del profesor, entregables comprimidos; la envía un solo integrante)
**Sustentación:** entre el 8 y el 12 de diciembre de 2026, **sin computador, solo papel y lápiz** (ver §11)
**Grupo (7, el mismo del video):** Sebas Alarcón · Edwin Corleone · Mateo Herrera · Angel Olivera ·
Sebastián Umaña · Alejandro Arguello · Nicolas Siabatto. Los Lineamientos (§3.1.5) fijan un máximo de 5;
el grupo lo gestionará con el profesor.

---

## 1. Título tentativo

**«Traductor probabilístico de órdenes en español a reglas de firewall mediante PCFG y CYK»**
(nombre corto de la herramienta: por definir, p. ej. *OrdenRed*)

## 2. Problema y justificación (borrador)

Configurar un firewall o una ACL exige una sintaxis rígida (iptables, nftables, Cisco IOS) y un error
deja un servicio caído o un hueco de seguridad. Un administrador piensa la regla en lenguaje natural
(«bloquea el SSH desde la red de invitados hacia el servidor web») y luego la traduce a mano.

La herramienta recibe la orden en español, la corrige, la clasifica, la analiza sintácticamente, construye
su significado y genera la regla, **mostrando el árbol, la probabilidad y las interpretaciones alternativas**
cuando la orden es ambigua. Es un problema de computación y comunicaciones, y su núcleo es probabilístico
(cadenas de Markov / n-gramas, Bayes, HMM, PCFG), que es justo el enfoque de la asignatura.

## 3. Cobertura del capítulo 24 (requisito: ≥ 50 % de la solución)

| Módulo | Técnica del capítulo | Sección | Desde cero |
|---|---|---|---|
| M1 Tokenizador | Tokenización; tokens especiales `<IP>`, `<CIDR>`, `<PUERTO>`, `<UNK>` | 24.1.1, 24.1.4 | Sí (regex de `re`) |
| M2 Corrector ortográfico | Modelo de n-gramas de caracteres y de palabras; suavizado de Laplace, *backoff* e interpolación lineal (λ ajustados en *held-out*); canal ruidoso | 24.1.2–24.1.4 | Sí |
| M3 Clasificador de intención | Bolsa de palabras + Naive Bayes (generativo) vs. regresión logística (discriminativo) | 24.1.1, 24.1.6 | Sí |
| M4 Etiquetador | HMM + algoritmo de Viterbi para asignar categoría a palabras fuera del léxico | 24.1.6 | Sí |
| M5 Analizador sintáctico | PCFG en FNC + CYK probabilístico; *k* mejores árboles para la ambigüedad | 24.2, 24.3 | Sí |
| M6 Aprendizaje de la gramática | Estimación de probabilidades de la PCFG por conteo sobre un *treebank* propio | 24.3.2 | Sí |
| M7 Intérprete semántico | Gramática aumentada con semántica composicional → forma lógica | 24.4, 24.4.1, 24.4.2 | Sí |
| M8 Generador de reglas | Forma lógica → **iptables** (salida principal) y **ACL extendida de Cisco IOS** (secundaria) | (no es del capítulo) | Sí |
| M9 Interfaz | CLI (`argparse`) + GUI tkinter (árbol, probabilidades, regla) | (no es del capítulo) | Sí |

M1–M7 son técnicas del capítulo y constituyen el grueso del código; M8–M9 son el «pegamento».

**Ejemplo de flujo:**
`"bloqea el puerto 22 desde 10.0.0.0/8"` → M2 corrige *bloqea → bloquea* → M3: intención = BLOQUEAR (p = 0.97)
→ M5 árbol más probable (p = 3.1e-6) → M7: `Acción(drop) ∧ Origen(10.0.0.0/8) ∧ Puerto(22) ∧ Proto(tcp)`
→ M8: `iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 22 -j DROP`
   y `access-list 101 deny tcp 10.0.0.0 0.255.255.255 any eq 22`

## 4. Datos

- **Corpus sintético** generado muestreando la propia PCFG (con plantillas, sinónimos y variación de orden)
  → *treebank* anotado con árbol, intención y forma lógica. División entrenamiento / validación / prueba.
- **Inyección de ruido** controlada (sustitución, omisión, transposición de letras) para el corrector.
- **Conjunto «humano»**: 100–200 órdenes escritas a mano por los integrantes, sin mirar la gramática,
  para medir cobertura real.
- Sin descargas externas.

## 5. Escenarios de prueba (mínimo 3; se proponen 5)

| # | Escenario | Qué se mide |
|---|---|---|
| E1 | Órdenes limpias dentro de la gramática | Exactitud de intención, exactitud del árbol, exactitud de la regla generada |
| E2 | Ruido tipográfico (0–20 % de caracteres) | Exactitud con y sin corrector; perplejidad con Laplace vs. *backoff* vs. interpolación |
| E3 | Ambigüedad estructural (adjunción de complementos: «desde…», «hacia…», «por…») | PCFG aprendida vs. probabilidades uniformes; posición de la interpretación correcta en los *k* mejores |
| E4 | Tamaño del corpus de entrenamiento y vocabulario desconocido | Curvas de aprendizaje; Naive Bayes vs. regresión logística; aporte del HMM a las palabras `<UNK>` |
| E5 | Escalabilidad de CYK | Tiempo vs. longitud de la orden y tamaño de la gramática (contraste con O(n³·\|G\|)) |

Cada escenario se corre con varias semillas y se reporta media e intervalo de confianza.

## 6. Diseño (lenguaje: UML)

- Diagrama de casos de uso (administrador: traducir orden, ver alternativas, exportar reglas, entrenar modelos).
- Diagrama de componentes / bloques del *pipeline* M1→M9.
- Diagrama de clases (`Tokenizador`, `ModeloNGramas`, `NaiveBayes`, `RegresionLogistica`, `HMM`, `PCFG`, `ParserCYK`, `InterpreteSemantico`, `GeneradorReglas`).
- Diagrama de secuencia de una traducción completa.
- Diagramas de actividad / pseudocódigo para CYK, Viterbi y el entrenamiento de la PCFG.

## 7. Stack técnico

- **Python 3.13** (instalación estándar) + **NumPy 2.x** como la librería de estructuras de datos
  (tablas de CYK, matrices de Viterbi, gradiente de la regresión logística).
- **matplotlib**: solo para las figuras de los experimentos (no interviene en la solución).
- La justificación de NumPy y matplotlib la redacta el grupo más adelante. Regla interna: **no usar
  ninguna función de NumPy que resuelva una parte del problema**; solo arreglos y operaciones aritméticas.
- **tkinter** para la GUI: hoy **no está instalado** en esta máquina (`sudo apt install python3-tk`).
- Pruebas con `unittest` (stdlib).
- `requirements.txt` con las versiones exactas de NumPy y matplotlib (lo exige el literal d).
- Documento: un solo PDF (LaTeX, `pdflatex` ya está instalado) + `.zip` con el código.

### Estructura propuesta del repositorio

```
Proyecto/
├── PLAN.md
├── src/ordenred/
│   ├── tokenizador.py   ngramas.py   corrector.py
│   ├── naive_bayes.py   regresion_logistica.py   hmm.py
│   ├── pcfg.py   cyk.py   semantica.py   generador_reglas.py
│   ├── cli.py   gui.py
│   └── gramatica/ (reglas.txt, lexico.txt)
├── datos/ (generador del corpus, treebank, conjunto humano)
├── experimentos/ (E1–E5, resultados .csv, figuras)
├── tests/
└── informe/ (LaTeX → informe.pdf)
```

## 8. Cronograma (14 días)

| Fechas | Hito |
|---|---|
| 23–24 nov | Asignar roles; gramática v1 y léxico; interfaces entre módulos acordadas |
| 25–28 nov | M1, M5, M6, M7, M8 (núcleo: orden limpia → regla) + generador de corpus |
| 28 nov–1 dic | M2, M3, M4; CLI; pruebas unitarias |
| 1–3 dic | GUI; experimentos E1–E5 |
| 2–5 dic | Informe: marco teórico, problema, diseño UML, manuales, análisis de escenarios |
| 4–6 dic | Guía de sustentación (§11) y simulacros en papel |
| 6 dic | Revisión final del grupo, empaquetado |
| 7 dic | Envío del correo de entrega |
| 8–12 dic | Sustentación (papel y lápiz) |

## 9. Riesgos y pendientes

1. **Trámites con el profesor (los hace el grupo más adelante):** excepción de uso de IA (§3.3.2),
   permiso para NumPy y matplotlib, y el grupo de 7 frente al máximo de 5 (§3.1.5).
2. **Alcance:** si el tiempo aprieta, se recortan M4 (HMM) y la regresión logística; el núcleo M1, M2, M3, M5, M6, M7 sigue cubriendo bien más del 50 %.
3. **Cobertura de la gramática** con órdenes humanas: se mitiga con `<UNK>` + HMM y con el conjunto humano para iterar.
4. **Sustentación sin computador:** cualquiera puede tener que ejecutar a mano un algoritmo del proyecto;
   todos deben dominar el *pipeline* completo, no solo su módulo (ver §11).

## 10. Roles (7)

El detalle completo (contratos entre módulos, tareas por día, hitos, reparto del informe, escenarios y
ejercicios de sustentación) está en [DIVISION_DE_TRABAJO.md](DIVISION_DE_TRABAJO.md).

| # | Rol | Módulos |
|---|---|---|
| 1 | Arquitectura sintáctica | M5 PCFG + CYK, M6 aprendizaje de la PCFG |
| 2 | Semántica y reglas | M7 intérprete semántico, M8 iptables y ACL |
| 3 | Modelos de lenguaje | M2 n-gramas, suavizado y corrector |
| 4 | Clasificación y etiquetado | M3 Naive Bayes y regresión logística, M4 HMM + Viterbi |
| 5 | Datos y pruebas | M1 tokenizador, corpus, conjunto humano, pruebas de integración |
| 6 | Integración e interfaz | Pipeline, CLI, GUI, UML, manuales |
| 7 | Experimentos e informe | E1–E5, PDF, correo, guía de sustentación |

## 11. Preparación de la sustentación (papel y lápiz)

Entregable interno (no va al profesor): **guía de sustentación** con ejercicios resueltos a mano, tomados
del propio sistema y con números pequeños verificados con el código.

| Tema | Ejercicio a mano |
|---|---|
| Tokenización | Tokenizar una orden con IP, CIDR y puerto; qué va a `<UNK>` |
| N-gramas y suavizado | Bigramas sobre 4–5 órdenes; probabilidad de una orden con Laplace y con interpolación (λ dados) |
| Corrector | Elegir la corrección más probable de «bloqea» por canal ruidoso |
| Naive Bayes | Clasificar una orden corta con una tabla de conteos de 3 intenciones |
| HMM / Viterbi | Tabla de Viterbi de 3 palabras × 3 etiquetas |
| PCFG / CYK | Tabla CYK de una orden de 4–5 palabras con una gramática reducida en FNC; elegir el árbol más probable |
| Ambigüedad | Dos árboles de una misma orden y cuál gana con la PCFG |
| Semántica | Construir la forma lógica a partir del árbol y la regla iptables / ACL resultante |
| Diseño | Dibujar de memoria el diagrama de componentes y la secuencia de una traducción |
| Resultados | Explicar las figuras de E1–E5 y por qué salen así |

Se hacen al menos dos simulacros en los que cada integrante resuelve en papel un ejercicio elegido al azar.
