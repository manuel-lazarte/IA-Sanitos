"""Pruebas de carga y validación de documentos (src/carga_datos.py)."""
from pathlib import Path

import pandas as pd
import pytest

from src.carga_datos import DatosInvalidosError, cargar_carpeta_txt, cargar_csv


def test_carga_csv_demo_completa(ruta_demo):
    df, informe = cargar_csv(ruta_demo)
    assert len(df) == 30
    assert list(df.columns) == ["titulo", "medio", "fecha", "url", "contenido"]
    assert informe["validos"] == 30
    assert informe["duplicados"] == 0
    assert informe["invalidos"] == 0


def test_carga_normaliza_encabezados_con_tildes(tmp_path):
    ruta = tmp_path / "noticia.csv"
    ruta.write_text(
        "Título,Medio,Fecha,URL,Contenido\n"
        "Nota de prueba,Medio Demo 1,2026-11-01,https://demo.ejemplo/x/001,"
        "Texto con suficientes palabras para ser un documento valido aqui.\n",
        encoding="utf-8",
    )
    df, _ = cargar_csv(ruta)
    assert len(df) == 1
    assert df.loc[0, "titulo"] == "Nota de prueba"


def test_columna_faltante_lanza_error_con_nombre(tmp_path):
    ruta = tmp_path / "incompleto.csv"
    ruta.write_text("titulo,medio,fecha,url\nA,Medio,2026-01-01,https://x\n", encoding="utf-8")
    with pytest.raises(DatosInvalidosError) as exc:
        cargar_csv(ruta)
    assert "contenido" in str(exc.value)


def test_invalidos_y_duplicados_contados(tmp_path):
    largo = "un texto largo con muchas palabras para pasar el minimo establecido"
    ruta = tmp_path / "sucio.csv"
    ruta.write_text(
        "titulo,medio,fecha,url,contenido\n"
        f"N1,Medio Demo 1,2026-11-01,https://demo.ejemplo/a/001,{largo}\n"
        "Corta,Medio Demo 1,2026-11-02,https://demo.ejemplo/a/002,muy poco\n"
        f"Duplicada,Medio Demo 2,2026-11-03,https://demo.ejemplo/a/003,{largo}\n",
        encoding="utf-8",
    )
    df, informe = cargar_csv(ruta)
    assert len(df) == 1
    assert informe["invalidos"] == 1
    assert informe["duplicados"] == 1
    assert informe["validos"] == 1


def test_carga_carpeta_txt(tmp_path):
    for n in range(3):
        (tmp_path / f"nota_{n}.txt").write_text(
            f"Cuerpo de la noticia numero {n} con bastantes palabras para ser valida.",
            encoding="utf-8",
        )
    df, informe = cargar_carpeta_txt(tmp_path)
    assert len(df) == 3
    assert set(df["medio"]) == {"Archivo TXT"}
    assert informe["validos"] == 3
