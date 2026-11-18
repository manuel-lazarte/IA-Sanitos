"""ETAPA 3 del agente: representación numérica del corpus - Grupo 15.

Dos rutas sobre el MISMO corpus preprocesado (informe 3.7 y 4.4):

Ruta A - LDA:  CountVectorizer produce una matriz documento-término de
  CONTEOS de palabras (bolsa de palabras). LDA es un modelo probabilístico
  generativo que trabaja con recuentos: cuántas veces aparece cada término.

Ruta B - NMF:  TfidfVectorizer produce pesos TF-IDF: la frecuencia local
  del término se pondera por su rareza en la colección, reduciendo la
  influencia de palabras presentes en casi todos los documentos.

Ambas rutas comparten PARAMS_VOCABULARIO para que sus vocabularios sean
comparables: la comparación de modelos exige representaciones homogéneas.
"""
from typing import Iterable

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix

# Vocabulario compartido: quita términos que aparecen en <2 documentos
# (ruido/hapax) o en >90% de ellos (demasiado generales), y acota el tamaño.
PARAMS_VOCABULARIO = {"min_df": 2, "max_df": 0.9, "max_features": 3000}


class CorpusInvalidoError(ValueError):
    """El corpus no deja vocabulario útil tras el preprocesamiento."""


def _vectorizar(
    textos: Iterable[str], fabrica
) -> tuple[CountVectorizer | TfidfVectorizer, csr_matrix]:
    textos = list(textos)
    vectorizer = fabrica(**PARAMS_VOCABULARIO)
    try:
        matriz = vectorizer.fit_transform(textos)
    except ValueError as error:
        # sklearn lanza ValueError cuando el vocabulario resultante es vacío.
        raise CorpusInvalidoError(
            "El corpus no produjo un vocabulario utilizable. Revisa el "
            "preprocesamiento: si todos los documentos quedaron vacíos "
            "(solo palabras vacías) o son demasiado parecidos, no hay "
            "señal temática. Detalle original: " + str(error)
        ) from error
    return vectorizer, matriz


def construir_matriz_conteos(textos: list[str]):
    """Ruta A: matriz de conteos para LDA. Devuelve (vectorizer, matriz)."""
    return _vectorizar(textos, CountVectorizer)


def construir_matriz_tfidf(textos: list[str]):
    """Ruta B: matriz TF-IDF para NMF. Devuelve (vectorizer, matriz)."""
    return _vectorizar(textos, TfidfVectorizer)


def terminos_vocabulario(vectorizer) -> list[str]:
    """Términos del vocabulario ajustado, en el orden de las columnas."""
    try:
        return list(vectorizer.get_feature_names_out())
    except NotFittedError as error:
        raise CorpusInvalidoError("El vectorizador aún no fue ajustado.") from error
