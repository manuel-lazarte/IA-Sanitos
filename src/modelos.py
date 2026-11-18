"""ETAPA 4 del agente: entrenamiento de los modelos de IA - Grupo 15.

MODELO 1 - LDA (Latent Dirichlet Allocation, Blei et al. 2003):
  Modelo probabilístico generativo. Cada tema es una distribución sobre el
  vocabulario y cada documento una mezcla de temas. Trabaja con CONTEOS
  (ruta A). `transform` devuelve las proporciones estimadas de temas por
  documento (filas que suman 1).

MODELO 2 - NMF (Non-negative Matrix Factorization, Kuang et al. 2015):
  Aproxima la matriz TF-IDF (ruta B) como producto de dos matrices no
  negativas. `transform` devuelve coeficientes de contribución por
  documento: NO son probabilidades; solo tras normalizar cada fila
  (`normalizar_pesos`) pueden leerse como proporciones de peso.

Reproducibilidad: SEMILLA_ALEATORIA fija en ambos modelos
(LDA batch, NMF nndsvda) para que los resultados sean comparables entre
ejecuciones (informe 4.4).
"""
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from scipy.sparse import spmatrix

SEMILLA_ALEATORIA = 42


class CorpusInsuficienteError(ValueError):
    """El corpus es demasiado pequeño para el número de temas pedido."""


def _validar_entrada(matriz: spmatrix, n_temas: int) -> None:
    if matriz is None or matriz.shape[0] == 0 or matriz.nnz == 0:
        raise CorpusInsuficienteError(
            "La matriz de documentos está vacía o sin términos: no hay nada "
            "que modelar. Revisa la carga y el preprocesamiento."
        )
    if n_temas < 2:
        raise CorpusInsuficienteError(
            "Se necesitan al menos 2 temas para el descubrimiento."
        )
    if matriz.shape[0] < n_temas:
        raise CorpusInsuficienteError(
            f"Corpus insuficiente: {matriz.shape[0]} documentos para "
            f"n_temas={n_temas}. Reduce el número de temas o agrega "
            "documentos (se recomienda al menos 3 documentos por tema)."
        )
    if matriz.shape[1] < n_temas:
        raise CorpusInsuficienteError(
            f"Vocabulario insuficiente: la matriz tiene solo {matriz.shape[1]} "
            f"términos útiles para n_temas={n_temas}. Esto pasa con corpus "
            "muy cortos o muy repetitivos: agrega más documentos variados o "
            "reduce el número de temas."
        )


def entrenar_lda(
    matriz: spmatrix,
    n_temas: int = 6,
    semilla: int = SEMILLA_ALEATORIA,
) -> LatentDirichletAllocation:
    """Entrena LDA sobre la matriz de conteos (ruta A)."""
    _validar_entrada(matriz, n_temas)
    modelo = LatentDirichletAllocation(
        n_components=n_temas,
        learning_method="batch",  # determinista con semilla en corpus pequeños
        max_iter=30,
        random_state=semilla,
    )
    modelo.fit(matriz)
    return modelo


def entrenar_nmf(
    matriz: spmatrix,
    n_temas: int = 6,
    semilla: int = SEMILLA_ALEATORIA,
) -> NMF:
    """Entrena NMF sobre la matriz TF-IDF (ruta B)."""
    _validar_entrada(matriz, n_temas)
    modelo = NMF(
        n_components=n_temas,
        init="nndsvda",  # estable y reproducible; no requiere semilla aleatoria
        max_iter=500,
        random_state=semilla,
    )
    modelo.fit(matriz)
    return modelo


def palabras_por_tema(modelo, vocabulario: list[str], n_palabras: int = 10) -> list[list[str]]:
    """Top-n términos por tema usando components_ (válido para LDA y NMF)."""
    componentes = modelo.components_
    return [
        [vocabulario[i] for i in componentes[k].argsort()[::-1][:n_palabras]]
        for k in range(componentes.shape[0])
    ]


def pesos_documento(modelo, matriz: spmatrix) -> np.ndarray:
    """Matriz (n_documentos, n_temas) de pertenencia temática.

    LDA: proporciones de mezcla (filas suman 1).
    NMF: coeficientes crudos de la factorización (no suman 1).
    """
    return modelo.transform(matriz)


def normalizar_pesos(pesos: np.ndarray) -> np.ndarray:
    """Convierte pesos (típicamente de NMF) en proporciones por documento.

    Únicamente para PRESENTACIÓN: los coeficientes de NMF no son
    probabilidades bajo un modelo generativo como sí lo son las
    proporciones de LDA (informe 3.11).
    """
    totales = pesos.sum(axis=1, keepdims=True)
    totales[totales == 0] = 1.0
    return pesos / totales


def tema_predominante(pesos: np.ndarray) -> np.ndarray:
    """Tema de mayor peso para cada documento (índice 0-based)."""
    return pesos.argmax(axis=1)


def indices_representativos(pesos: np.ndarray, n: int = 3) -> list[list[int]]:
    """Índices de los n documentos más representativos de cada tema."""
    return [list(pesos[:, k].argsort()[::-1][:n]) for k in range(pesos.shape[1])]
