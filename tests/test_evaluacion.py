"""Pruebas de evaluación temática (src/evaluacion.py).

Diversidad = proporción de términos únicos entre listas principales
(informe 3.13). Coherencia = NPMI media de las palabras principales,
calculada sobre el propio corpus, la MISMA medida para LDA y NMF.
"""
import numpy as np
import pytest

from src.evaluacion import (
    coherencia_npmi,
    conteo_predominante,
    diversidad_tematica,
    resumen_modelo,
)
from src.modelos import entrenar_lda, pesos_documento
from src.preproceso import procesar_corpus
from src.representacion import construir_matriz_conteos


def test_diversidad_temas_disjuntos():
    assert diversidad_tematica([["a", "b", "c"], ["d", "e", "f"]]) == 1.0


def test_diversidad_temas_identicos():
    assert diversidad_tematica([["a", "b"], ["a", "b"]]) == 0.5


def test_diversidad_vacia_es_cero():
    assert diversidad_tematica([]) == 0.0


def test_coherencia_tema_coherente_supera_a_cruzada(corpus_mini):
    """Palabras que co-ocurren en el corpus (temas reales) deben superar
    a pares cruzados de temas distintos (diésel con gol, etc.)."""
    textos = procesar_corpus(corpus_mini)
    temas_reales = [
        ["diésel", "combustible", "precio", "transporte"],
        ["gol", "equipo", "partido", "torneo"],
    ]
    temas_cruzadas = [
        ["diésel", "gol", "combustible", "equipo"],
        ["partido", "precio", "torneo", "transporte"],
    ]
    coherente = coherencia_npmi(temas_reales, textos, n_palabras=4)
    cruzada = coherencia_npmi(temas_cruzadas, textos, n_palabras=4)
    assert coherente > cruzada
    assert -1.0 <= coherente <= 1.0


def test_conteo_predominante_suma_documentos(mini_lda_pesos):
    conteo = conteo_predominante(mini_lda_pesos)
    assert sum(conteo.values()) == 8
    assert all(t >= 0 for t in conteo)


@pytest.fixture
def mini_lda_pesos(corpus_mini):
    textos = procesar_corpus(corpus_mini)
    vec, matriz = construir_matriz_conteos(textos)
    lda = entrenar_lda(matriz, n_temas=3)
    return pesos_documento(lda, matriz)


def test_resumen_modelo_claves_completas(mini_lda_pesos, corpus_mini):
    textos = procesar_corpus(corpus_mini)
    vec, matriz = construir_matriz_conteos(textos)
    vocabulario = list(vec.get_feature_names_out())
    lda = entrenar_lda(matriz, n_temas=3)
    resumen = resumen_modelo("LDA", lda, matriz, vocabulario, textos, n_palabras=5)
    assert set(resumen) == {
        "nombre", "n_temas", "temas", "diversidad", "coherencia", "distribucion",
    }
    assert resumen["nombre"] == "LDA"
    assert resumen["n_temas"] == 3
    assert len(resumen["temas"]) == 3
    primer = resumen["temas"][0]
    assert set(primer) == {"tema", "palabras", "coherencia"}
    assert len(primer["palabras"]) == 5
    assert sum(resumen["distribucion"].values()) == 8
    assert 0.0 <= resumen["diversidad"] <= 1.0
