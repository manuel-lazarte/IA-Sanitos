"""Pruebas del corpus REAL (DATASET_NOTICIAS_BOLIVIA_GRUPO15).

Se activan solo cuando el dataset fue recopilado (existe el CSV);
mientras tanto se marcan como salteadas. Verifican que el corpus real
sea consumible por el cargador del proyecto SIN modificarlo.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "DATASET_NOTICIAS_BOLIVIA_GRUPO15"
RUTA_CSV = DATASET / "noticias_bolivia.csv"

pytestmark = pytest.mark.skipif(
    not RUTA_CSV.exists(), reason="el corpus real aún no fue recopilado"
)

if RUTA_CSV.exists():
    sys.path.insert(0, str(ROOT))


def test_dataset_carga_con_el_cargador_del_proyecto():
    from src.carga_datos import cargar_csv

    df, informe = cargar_csv(RUTA_CSV)
    assert informe["validos"] == len(df)
    assert informe["duplicados"] == 0, "el corpus entregado no debe tener duplicados exactos"


def test_dataset_dentro_del_rango_del_informe():
    """Informe 4.3 / Tabla 1: entre 200 y 400 noticias, máximo dos medios."""
    import pandas as pd

    df = pd.read_csv(RUTA_CSV)
    assert 200 <= len(df) <= 400, f"{len(df)} registros fuera del rango del informe"
    assert df["medio"].nunique() <= 2


def test_dataset_campos_completos_y_fechas_iso():
    import pandas as pd

    df = pd.read_csv(RUTA_CSV)
    assert df["titulo"].notna().all() and df["contenido"].notna().all()
    assert df["fecha"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    assert df["url"].astype(str).str.startswith("http").all()
    assert df["contenido"].str.split().str.len().min() >= 30


def test_dataset_manifiesto_coincide():
    import pandas as pd

    manifiesto = pd.read_csv(DATASET / "trazabilidad_manifiesto.csv")
    df = pd.read_csv(RUTA_CSV)
    assert len(manifiesto) == len(df)
    assert manifiesto["id_noticia"].is_unique
    assert manifiesto["url"].is_unique
