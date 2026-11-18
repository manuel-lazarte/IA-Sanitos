"""Pruebas de las representaciones numéricas (src/representacion.py).

Ruta A - LDA: conteos de palabras (CountVectorizer).
Ruta B - NMF: pesos TF-IDF (TfidfVectorizer).
Ambas sobre el mismo corpus preprocesado con el MISMO vocabulario
(parámetros idénticos) para que la comparación sea justa (informe 4.4).
"""
import pytest

from src.preproceso import procesar_corpus
from src.representacion import (
    CorpusInvalidoError,
    construir_matriz_conteos,
    construir_matriz_tfidf,
    terminos_vocabulario,
)


@pytest.fixture
def textos_procesados_mini(corpus_mini):
    return procesar_corpus(corpus_mini)


def test_matriz_conteos_shape_y_enteros(textos_procesados_mini):
    vectorizer, matriz = construir_matriz_conteos(textos_procesados_mini)
    vocabulario = terminos_vocabulario(vectorizer)
    assert matriz.shape == (8, len(vocabulario))
    assert len(vocabulario) >= 5
    assert (matriz.data >= 0).all()
    # Conteos enteros: la entrada de LDA.
    assert float(matriz.sum()) == int(matriz.sum())


def test_matriz_tfidf_valores_ponderados(textos_procesados_mini):
    vectorizer, matriz = construir_matriz_tfidf(textos_procesados_mini)
    vocabulario = terminos_vocabulario(vectorizer)
    assert matriz.shape == (8, len(vocabulario))
    assert matriz.data.min() > 0
    assert matriz.data.max() <= 1.0  # normalización L2 por documento


def test_vocabularios_comparables_entre_rutas(textos_procesados_mini):
    vec_conteo, _ = construir_matriz_conteos(textos_procesados_mini)
    vec_tfidf, _ = construir_matriz_tfidf(textos_procesados_mini)
    assert set(terminos_vocabulario(vec_conteo)) == set(terminos_vocabulario(vec_tfidf))


def test_corpus_de_un_solo_termino_es_valido():
    # 'palabra' en 2 de 3 documentos (67% < max_df=0.9): vocabulario de 1 término.
    _, matriz = construir_matriz_conteos(["palabra", "palabra", "otra cosa"])
    assert matriz.shape == (3, 1)


def test_termino_en_todos_los_documentos_se_poda():
    """Un término presente en el 100% del corpus no aporta señal temática
    y max_df=0.9 lo poda; si es el único término, el corpus queda inválido."""
    with pytest.raises(CorpusInvalidoError):
        construir_matriz_conteos(["palabra", "palabra", "palabra"])


def test_corpus_sin_vocabulario_util_lanza_error():
    """Documentos procesados vacíos (todo eran stopwords) => error accionable."""
    with pytest.raises(CorpusInvalidoError) as exc:
        construir_matriz_conteos(["", "", ""])
    assert "vocabulario" in str(exc.value).lower()
    with pytest.raises(CorpusInvalidoError):
        construir_matriz_tfidf(["", "", ""])
