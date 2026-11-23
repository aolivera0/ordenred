"""Pruebas. Ejecutar desde la raíz del proyecto:  python -m unittest -v"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FIXTURES = RAIZ / "tests" / "fixtures"
sys.path.insert(0, str(RAIZ / "src"))
