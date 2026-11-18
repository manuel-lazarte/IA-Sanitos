"""Pruebas del corpus sintético de demostración (datos/noticias_demo.csv).

El corpus es 100% sintético: medios ficticios, textos generados con
semilla fija. Estas pruebas fijan su estructura y su vocabulario temático.
"""
import pandas as pd


def test_csv_demo_estructura(ruta_demo):
    df = pd.read_csv(ruta_demo)
    assert list(df.columns) == ["titulo", "medio", "fecha", "url", "contenido"]
    assert len(df) == 30
    assert set(df["medio"]) <= {"Medio Demo 1", "Medio Demo 2"}
    assert df["contenido"].str.split().str.len().min() >= 30
    assert df["contenido"].nunique() == 30
    assert df["url"].str.startswith("https://demo.ejemplo/").all()


def test_csv_demo_vocabulario_tematico(ruta_demo):
    texto = " ".join(pd.read_csv(ruta_demo)["contenido"]).lower()
    for palabra in [
        "combustible", "diésel", "gasolina", "sequía", "precio",
        "dólar", "partido", "vacuna", "estudiante", "escuela",
    ]:
        assert palabra in texto, f"falta '{palabra}' en el corpus demo"
