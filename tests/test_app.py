"""Pruebas de los helpers de presentación de la app (app.py).

La UI completa se verifica manualmente con `streamlit run app.py`;
aquí se prueban las funciones puras que convierten los resultados de
los modelos en tablas y etiquetas para mostrar (ETAPA 5, presentación).
"""
import pandas as pd
import pytest

import app


def test_importar_app_no_levanta_la_interfaz():
    """Importar el módulo no debe ejecutar la UI (guard __main__)."""
    assert callable(app.main)


def test_tabla_temas_columnas_y_filas():
    resumen = {
        "nombre": "NMF",
        "n_temas": 2,
        "temas": [
            {"tema": 0, "palabras": ["a", "b", "c"], "coherencia": 0.25},
            {"tema": 1, "palabras": ["d", "e", "f"], "coherencia": -0.1},
        ],
        "diversidad": 1.0,
        "coherencia": 0.075,
        "distribucion": {0: 5, 1: 4},
    }
    tabla = app.tabla_temas(resumen)
    assert isinstance(tabla, pd.DataFrame)
    assert list(tabla.columns) == ["Tema", "Palabras", "Noticias predominantes", "Coherencia"]
    assert len(tabla) == 2
    assert tabla.loc[0, "Tema"] == 1  # 1-based para mostrar
    assert tabla.loc[1, "Tema"] == 2
    assert tabla.loc[0, "Noticias predominantes"] == 5
    assert tabla.loc[1, "Coherencia"] == pytest.approx(-0.1)


def test_etiqueta_tema_es_1_based_y_acota_palabras():
    etiqueta = app.etiqueta_tema(2, ["a", "b", "c", "d", "e"], 3)
    assert etiqueta == "Tema 3 · a, b, c"


def test_etiqueta_tema_con_menos_palabras_que_n():
    etiqueta = app.etiqueta_tema(0, ["x", "y"], 5)
    assert etiqueta == "Tema 1 · x, y"


def test_etiquetas_eje_respetan_ids_de_temas():
    """Si un tema nunca predomina (sin clave en el dict), los demás NO se
    renumeran: {0: 12, 2: 18} debe etiquetarse T1 y T3, nunca T1 y T2."""
    assert app.etiquetas_temas({0: 12, 2: 18}) == ["T1", "T3"]
    assert app.etiquetas_temas({}) == []
