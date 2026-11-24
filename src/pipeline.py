"""Orquestador del agente: une las seis etapas del informe - Grupo 15.

`ejecutar_pipeline` es el punto único de entrada usado por la interfaz
Streamlit y por las pruebas: recibe el DataFrame validado de noticias y
devuelve un diccionario con todos los artefactos del análisis:

  {
    "df":                DataFrame original (trazabilidad),
    "textos_procesados": cuerpo de cada noticia tras la ETAPA 2,
    "conteos":           (vectorizer, matriz) ruta A -> LDA,
    "tfidf":             (vectorizer, matriz) ruta B -> NMF,
    "lda":               modelo LatentDirichletAllocation entrenado,
    "nmf":               modelo NMF entrenado,
    "resumenes":         {"LDA": resumen, "NMF": resumen} de evaluacion,
  }

No contiene lógica propia: cada etapa vive en su módulo independiente.
"""
import pandas as pd

from src.evaluacion import resumen_modelo
from src.modelos import (
    SEMILLA_ALEATORIA,
    CorpusInsuficienteError,
    entrenar_lda,
    entrenar_nmf,
)
from src.preproceso import procesar_corpus
from src.representacion import (
    construir_matriz_conteos,
    construir_matriz_tfidf,
    terminos_vocabulario,
)


def _validar_corpus(df: pd.DataFrame, n_temas: int) -> None:
    """Falla pronto (antes de spaCy) si no hay documentos suficientes."""
    n_docs = 0 if df is None else len(df)
    if n_docs == 0:
        raise CorpusInsuficienteError(
            "No hay documentos cargados: carga un CSV o TXT con noticias."
        )
    if n_docs < n_temas:
        raise CorpusInsuficienteError(
            f"Corpus insuficiente: {n_docs} documentos para n_temas={n_temas}. "
            "Reduce el número de temas o agrega documentos."
        )


def ejecutar_pipeline(
    df: pd.DataFrame,
    n_temas: int = 6,
    n_palabras: int = 10,
    semilla: int = SEMILLA_ALEATORIA,
    textos_procesados: list[str] | None = None,
) -> dict:
    """Ejecuta preprocesamiento, representación, modelos y evaluación.

    Si la interfaz ya preprocesó los textos (página 2), puede pasarlos en
    `textos_procesados` para no repetir la lematización con spaCy.
    """
    _validar_corpus(df, n_temas)

    if textos_procesados is None:
        textos = df["contenido"].fillna("").astype(str).tolist()
        textos_procesados = procesar_corpus(textos)

    # Ruta A: conteos para LDA. Ruta B: TF-IDF para NMF (mismo corpus,
    # vocabulario comparable por parámetros compartidos).
    vec_conteo, matriz_conteos = construir_matriz_conteos(textos_procesados)
    vec_tfidf, matriz_tfidf = construir_matriz_tfidf(textos_procesados)

    lda = entrenar_lda(matriz_conteos, n_temas=n_temas, semilla=semilla)
    nmf = entrenar_nmf(matriz_tfidf, n_temas=n_temas, semilla=semilla)

    resumenes = {
        "LDA": resumen_modelo(
            "LDA", lda, matriz_conteos, terminos_vocabulario(vec_conteo),
            textos_procesados, n_palabras,
        ),
        "NMF": resumen_modelo(
            "NMF", nmf, matriz_tfidf, terminos_vocabulario(vec_tfidf),
            textos_procesados, n_palabras,
        ),
    }

    return {
        "df": df,
        "textos_procesados": textos_procesados,
        "conteos": (vec_conteo, matriz_conteos),
        "tfidf": (vec_tfidf, matriz_tfidf),
        "lda": lda,
        "nmf": nmf,
        "resumenes": resumenes,
    }
