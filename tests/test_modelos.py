"""Pruebas del entrenamiento de LDA y NMF (src/modelos.py).

Ambos modelos reales de scikit-learn sobre corpus_mini procesado.
Verifica reproducibilidad con semilla fija y el error de corpus
insuficiente, y que NMF produce pesos no negativos (coeficientes de
la factorización, no probabilidades — informe 3.11).
"""
import numpy as np
import pytest

from src.modelos import (
    CorpusInsuficienteError,
    SEMILLA_ALEATORIA,
    entrenar_lda,
    entrenar_nmf,
    indices_representativos,
    palabras_por_tema,
    pesos_documento,
    tema_predominante,
)
from src.preproceso import procesar_corpus
from src.representacion import construir_matriz_conteos, construir_matriz_tfidf


@pytest.fixture
def mini_preparado(corpus_mini):
    textos = procesar_corpus(corpus_mini)
    vec_conteo, conteos = construir_matriz_conteos(textos)
    vec_tfidf, tfidf = construir_matriz_tfidf(textos)
    vocabulario = list(vec_conteo.get_feature_names_out())
    return conteos, tfidf, vocabulario


def test_lda_entrena_y_distribucion_normalizada(mini_preparado):
    conteos, _, _ = mini_preparado
    lda = entrenar_lda(conteos, n_temas=3)
    pesos = pesos_documento(lda, conteos)
    assert pesos.shape == (8, 3)
    assert np.allclose(pesos.sum(axis=1), 1.0, atol=1e-6)  # proporciones de mezcla


def test_nmf_entrena_con_pesos_no_negativos(mini_preparado):
    _, tfidf, _ = mini_preparado
    nmf = entrenar_nmf(tfidf, n_temas=3)
    assert (nmf.components_ >= 0).all()  # factorización no negativa
    pesos = pesos_documento(nmf, tfidf)
    assert pesos.shape == (8, 3)
    assert pesos.min() >= 0
    # Los pesos crudos de NMF NO suman 1 (no son probabilidades).
    # La normalización a proporciones es una decisión de presentación aparte.


def test_palabras_por_tema_validas(mini_preparado):
    conteos, _, vocabulario = mini_preparado
    lda = entrenar_lda(conteos, n_temas=3)
    temas = palabras_por_tema(lda, vocabulario, n_palabras=3)
    assert len(temas) == 3
    assert all(len(t) == 3 for t in temas)
    assert all(p in vocabulario for t in temas for p in t)
    assert all(len(set(t)) == len(t) for t in temas)  # sin repeticiones internas


def test_tema_predominante_en_rango(mini_preparado):
    conteos, _, _ = mini_preparado
    lda = entrenar_lda(conteos, n_temas=3)
    predominante = tema_predominante(pesos_documento(lda, conteos))
    assert predominante.shape == (8,)
    assert ((predominante >= 0) & (predominante < 3)).all()


def test_indices_representativos_validos(mini_preparado):
    _, tfidf, _ = mini_preparado
    nmf = entrenar_nmf(tfidf, n_temas=3)
    indices = indices_representativos(pesos_documento(nmf, tfidf), n=2)
    assert len(indices) == 3
    assert all(len(i) == 2 for i in indices)
    assert all(0 <= i < 8 for i in indices for i in i)


def test_mas_temas_que_documentos_lanza_error(mini_preparado):
    conteos, _, _ = mini_preparado
    with pytest.raises(CorpusInsuficienteError) as exc:
        entrenar_lda(conteos, n_temas=10)  # solo 8 documentos
    assert "n_temas" in str(exc.value)


def test_reproducibilidad_con_misma_semilla(mini_preparado):
    conteos, _, vocabulario = mini_preparado
    lda1 = entrenar_lda(conteos, n_temas=3, semilla=SEMILLA_ALEATORIA)
    lda2 = entrenar_lda(conteos, n_temas=3, semilla=SEMILLA_ALEATORIA)
    assert palabras_por_tema(lda1, vocabulario) == palabras_por_tema(lda2, vocabulario)
