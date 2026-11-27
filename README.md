# OrdenRed

Traductor probabilístico de órdenes en español a reglas de firewall (iptables y ACL de Cisco).
Proyecto de curso de *Modelos Estocásticos y Simulación en Computación y Comunicaciones* (UNAL),
basado en el capítulo 24 (*Natural Language Processing*) de Russell y Norvig, AIMA 4.ª ed. global.

- Plan general: [PLAN.md](PLAN.md)
- Roles, contratos entre módulos y cronograma: [DIVISION_DE_TRABAJO.md](DIVISION_DE_TRABAJO.md)

## Requisitos

- Python 3.13
- `pip install -r requirements.txt` (NumPy y matplotlib)
- Para la GUI: `sudo apt install python3-tk`

## Pruebas

Desde esta carpeta:

```bash
python3 -m unittest -v
```

### Con Docker (mismo entorno Linux en cualquier sistema)

Requiere Docker con `docker compose`. La imagen trae Python 3.13 y las versiones exactas de
`requirements.txt`; el código se monta en vivo, así que solo hay que reconstruir si cambia `requirements.txt`.

```bash
docker compose build                     # una vez
docker compose run --rm ordenred         # corre todas las pruebas
docker compose run --rm ordenred python experimentos/e4_clasificacion_etiquetado.py --fixtures --salida experimentos/resultados   # cualquier otro comando
```

El contenedor se borra al terminar (`--rm`): solo sobrevive lo que se escribe dentro del repositorio montado
en `/app` (p. ej. `experimentos/resultados/`); lo que quede en `/tmp` del contenedor se pierde.

En Git Bash (Windows), anteponga `MSYS_NO_PATHCONV=1` a los comandos que pasan rutas absolutas del
contenedor (como `/tmp/e4`); si no, Git Bash las convierte en rutas de Windows y la salida termina dentro del repo.

En Linux, si su usuario no tiene el uid 1000, construya la imagen con su uid para que los archivos creados en
el montaje le pertenezcan: `docker compose build --build-arg UID=$(id -u)`.

`tests/test_contratos.py` lista en `PENDIENTES` los métodos que aún son *stubs*. Al implementar uno,
**quítelo de la lista**: la prueba falla si un método implementado sigue listado.

## Uso (cuando esté implementado)

```bash
PYTHONPATH=src python3 -m ordenred.cli traducir "bloquea el ssh desde 10.0.0.0/8" --formato ambos --arbol
PYTHONPATH=src python3 -m ordenred.gui
```

## Estructura

| Ruta | Contenido | Dueño |
|---|---|---|
| `src/ordenred/tokenizador.py` | M1 · `Token`, `tokenizar` | Rol 5 |
| `src/ordenred/ngramas.py`, `corrector.py` | M2 · n-gramas, suavizado, canal ruidoso | Rol 3 |
| `src/ordenred/clasificadores.py` | M3 · Naive Bayes, regresión logística | Rol 4 |
| `src/ordenred/hmm.py` | M4 · HMM + Viterbi | Rol 4 |
| `src/ordenred/arbol.py`, `pcfg.py`, `cyk.py` | M5/M6 · árbol, PCFG, CYK | Rol 1 |
| `src/ordenred/semantica.py`, `objetos.py`, `generador_reglas.py` | M7/M8 · forma lógica, iptables, ACL | Rol 2 |
| `src/ordenred/corpus.py` | Carga y validación del corpus (C5) | Rol 5 |
| `src/ordenred/pipeline.py`, `cli.py`, `gui.py` | Integración e interfaz | Rol 6 |
| `src/ordenred/gramatica/` | `reglas.pcfg` (rol 1), `objetos.json` (rol 2) | Roles 1 y 2 |
| `datos/` | Generador de corpus, ruido, `*.jsonl` | Rol 5 |
| `tests/fixtures/` | Mini gramática, mini corpus y resultados esperados del CYK | Roles 1 y 5 |
| `experimentos/` | Runner de E1–E5, resultados y figuras | Rol 7 |
| `diseno/` | Diagramas UML | Rol 6 |
| `informe/` | LaTeX del PDF único y guía de sustentación | Rol 7 |
| `modelos/` | Modelos entrenados (no se versionan) | — |

## Fixtures disponibles desde el primer día

- `tests/fixtures/mini.pcfg`: gramática de 29 reglas en el formato C2, con una ambigüedad de adjunción a propósito.
- `tests/fixtures/mini_corpus.jsonl`: 5 órdenes con árbol, etiquetas, formas lógicas y reglas de oro (C5).
- `tests/fixtures/mini_cyk_esperado.json`: número de árboles y sus probabilidades para cada orden del mini
  corpus, calculados por enumeración exhaustiva. Sirve para probar el CYK y los *k* mejores árboles
  (ojo: la segunda orden tiene empates).
- `tests/fixtures/mini_objetos.json`: servicios y redes con nombre.

## Reglas

Solo stdlib + NumPy (solo arreglos y aritmética) + matplotlib (solo figuras). Todo algoritmo del capítulo
se implementa desde cero y cita la sección del libro en su *docstring*. Ver DIVISION_DE_TRABAJO.md §1.
