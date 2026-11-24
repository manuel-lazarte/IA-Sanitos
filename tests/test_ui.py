"""Pruebas de la capa de interfaz: adaptadores, filtros y componentes HTML.

La UI se verifica manualmente con `streamlit run app.py`; aquí se prueban
las funciones puras que leen los resultados (LDA/NMF/BERTopic), los filtros
y los fragmentos HTML, para que no se inventen cifras ni se inyecte HTML
desde los datos.
"""
import numpy as np
import pandas as pd
import pytest

import app
from ui import adaptadores, componentes


@pytest.fixture
def df_prueba() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "titulo": ["Noticia uno", "Noticia dos", "Noticia tres"],
            "medio": ["El Diario", "Prensa Libre", "El Diario"],
            "fecha": ["2026-01-01", "2026-01-02", "2026-01-02"],
            "url": ["http://a", "http://b", ""],
            "contenido": ["texto de la una", "diésel y gasolina", "otro contenido"],
        }
    )


class _VectorizadorFalso:
    def get_feature_names_out(self):
        return np.array(["combustible", "diésel", "gasolina"])


@pytest.fixture
def pipeline_falso(df_prueba: pd.DataFrame) -> dict:
    """Pipeline mínimo con el MISMO contrato que src.pipeline.ejecutar_pipeline.

    Los modelos se simulan con transform() y pesos conocidos para poder
    verificar quién es representativo sin reentrenar nada.
    """
    pesos = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    resumen = {
        "nombre": "LDA",
        "n_temas": 2,
        "temas": [
            {"tema": 0, "palabras": ["combustible", "diésel", "gasolina"], "coherencia": 0.1},
            {"tema": 1, "palabras": ["deporte", "partido", "equipo"], "coherencia": 0.2},
        ],
        "diversidad": 1.0,
        "coherencia": 0.15,
        "distribucion": {0: 2, 1: 1},
    }

    class _ModeloFalso:
        components_ = np.array([[5.0, 3.0, 1.0], [1.0, 4.0, 2.0]])

        def transform(self, _matriz):
            return pesos

    return {
        "df": df_prueba,
        "textos_procesados": ["uno", "dos", "tres"],
        "conteos": (_VectorizadorFalso(), object()),
        "tfidf": (_VectorizadorFalso(), object()),
        "lda": _ModeloFalso(),
        "nmf": _ModeloFalso(),
        "resumenes": {"LDA": resumen, "NMF": dict(resumen, nombre="NMF")},
    }


# ------------------------------------------------------------- adaptadores
def test_cargar_resultados_lda_estructura_comun(pipeline_falso):
    salida = adaptadores.cargar_resultados(pipeline_falso, "LDA")
    assert salida["metodo"] == "LDA"
    assert salida["resumen"]["n_temas"] == 2
    assert salida["pesos"].shape == (3, 2)
    assert salida["proporciones"].shape == (3, 2)
    # Las proporciones de presentación suman 1 por noticia.
    np.testing.assert_allclose(salida["proporciones"].sum(axis=1), 1.0)
    assert salida["vocabulario"] == ["combustible", "diésel", "gasolina"]
    assert salida["componentes"].shape == (2, 3)


def test_cargar_resultados_nmf_usa_tfidf(pipeline_falso):
    salida = adaptadores.cargar_resultados(pipeline_falso, "NMF")
    assert salida["metodo"] == "NMF"
    assert salida["resumen"]["nombre"] == "NMF"


def test_bertopic_no_tiene_funcionalidad(pipeline_falso):
    """BERTopic es el único método sin conexión: siempre None (sin cifras)."""
    assert adaptadores.cargar_resultados(pipeline_falso, "BERTopic") is None
    assert adaptadores.obtener_resultados_bertopic(pipeline_falso) is None


def test_metodo_desconocido_no_cuelga_datos_de_otro(pipeline_falso):
    assert adaptadores.cargar_resultados(pipeline_falso, "XYZ") is None


def test_conteo_por_tema_cuenta_solo_filtradas(pipeline_falso):
    res = adaptadores.cargar_resultados(pipeline_falso, "LDA")
    todos = np.array([0, 1, 2])
    assert adaptadores.conteo_por_tema(res["pesos"], todos) == [2, 1]
    # Doc 1 (tema 1 predominante) fuera del filtro: su tema queda en 0.
    assert adaptadores.conteo_por_tema(res["pesos"], np.array([0, 2])) == [2, 0]
    assert adaptadores.conteo_por_tema(res["pesos"], np.array([], dtype=int)) == [0, 0]


def test_noticias_representativas_ordena_por_peso(pipeline_falso):
    res = adaptadores.cargar_resultados(pipeline_falso, "LDA")
    reps = adaptadores.noticias_representativas(res, np.array([0, 1, 2]), tema=0, n=3)
    # Pesos del tema 0: doc0=0.9, doc1=0.2, doc2=0.6 -> orden 0, 2, 1.
    assert [i for i, _ in reps] == [0, 2, 1]
    assert reps[0][1] == pytest.approx(0.9)
    assert adaptadores.noticias_representativas(
        res, np.array([], dtype=int), tema=0
    ) == []


def test_pesos_palabras_toma_top_n(pipeline_falso):
    res = adaptadores.cargar_resultados(pipeline_falso, "LDA")
    top = adaptadores.pesos_palabras(res, tema=0, n=2)
    assert list(top) == ["combustible", "diésel"]  # componentes [5, 3, 1]
    assert top["combustible"] == pytest.approx(5.0)


# ------------------------------------------------------------------ filtros
def test_filtrar_sin_filtros_deja_todo(df_prueba):
    assert adaptadores.filtrar_noticias(df_prueba).tolist() == [0, 1, 2]


def test_filtrar_por_medio_y_fecha(df_prueba):
    assert adaptadores.filtrar_noticias(
        df_prueba, medios=["El Diario"]
    ).tolist() == [0, 2]
    assert adaptadores.filtrar_noticias(
        df_prueba, rango=("2026-01-02", "2026-01-02")
    ).tolist() == [1, 2]


def test_filtrar_por_texto_no_distingue_mayusculas_ni_tildes(df_prueba):
    # «diesel» (sin tilde, mayúsculas) encuentra «diésel y gasolina».
    assert adaptadores.filtrar_noticias(df_prueba, texto="DIESEL").tolist() == [1]
    assert adaptadores.filtrar_noticias(df_prueba, texto="no-existe").tolist() == []


# ------------------------------------------------- componentes (HTML puro)
def test_stepper_html_marca_paso_actual():
    salida = componentes.stepper_html(2)
    assert salida.count('class="step ') == 3  # los 3 pasos (sin contar el nav)
    assert 'class="step done"' in salida  # paso 1 completado
    assert 'class="step on"' in salida  # paso 2 actual
    assert "✓" in salida


def test_chips_html_escapa_datos():
    salida = componentes.chips_html(["<script>", "ok"])
    assert "<script>" not in salida
    assert "&lt;script&gt;" in salida
    assert ">ok</span>" in salida


def test_etiqueta_metodo_html_usa_color_fijo():
    assert "#1F5FB0" in componentes.etiqueta_metodo_html("LDA")
    assert "#B8461A" in componentes.etiqueta_metodo_html("NMF")
    # Método sin color propio: color neutro, nunca inventa uno.
    assert "#1B1F24" in componentes.etiqueta_metodo_html("XYZ")


def test_noticia_html_sin_url_no_muestra_enlace():
    salida = componentes.noticia_html("Título", "Medio", "2026-01-01", "", "extracto")
    assert "Abrir enlace" not in salida
    assert "extracto" in salida


def test_noticia_html_con_url_y_peso():
    salida = componentes.noticia_html(
        "Título", "Medio", "2026-01-01", "https://x", "extracto", 0.42
    )
    assert 'href="https://x"' in salida
    assert "42%" in salida


def test_noticia_html_escapa_titulo_malo():
    salida = componentes.noticia_html(
        "<img onerror=x>", "M", "2026-01-01", "", "e"
    )
    assert "<img" not in salida


# ------------------------------------------------------- helpers de app.py
def test_fmt_no_inventa_cifras():
    assert app._fmt(None) == "—"
    assert app._fmt(float("nan")) == "—"
    assert app._fmt(0.123456) == "0.123"
    assert app._fmt(1.5, 2) == "1.50"


def test_nombre_sugerido_es_derivado_no_inventado():
    assert app.nombre_sugerido(["combustible", "diésel"]) == "combustible, diésel"
    assert app.nombre_sugerido([]) == "—"


def test_txt_y_extracto_cuando_falta_dato():
    assert app._txt(None) == "—"
    assert app._txt(float("nan")) == "—"
    assert app._txt("  hola  ") == "hola"
    assert app._extracto(None) == "—"
    assert app._extracto("x" * 300).endswith("…")
    assert len(app._extracto("x" * 300)) == app.EXTRACTO_MAX + 1


def test_tabla_comparativa_muestra_bertopic_como_no_disponible(pipeline_falso):
    tabla = app.tabla_comparativa(pipeline_falso)
    assert list(tabla.columns) == ["LDA", "NMF", "BERTopic"]
    assert tabla.loc["Estado", "LDA"] == "Disponible"
    assert tabla.loc["Estado", "BERTopic"] == "Aún no disponible"
    assert tabla.loc["Coherencia NPMI", "BERTopic"] == "—"
    assert tabla.loc["Nº de temas", "LDA"] == "2"  # texto: tipos homogéneos
