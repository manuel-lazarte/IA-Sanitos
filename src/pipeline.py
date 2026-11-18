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
from src.modelos import SEMILLA_ALEATORIA, entrenar_lda, entrenar_nmf
from src.preproceso import procesar_corpus
from src.representacion import construir_matriz_conteos, construir_matriz_tfidf


def ejecutar_pipeline(
    df: pd.DataFrame,
    n_temas: int = 6,
    n_palabras: int = 10,
    semilla: int = SEMILLA_ALEATORIA,
) -> dict:
    """Ejecuta preprocesamiento, representación, modelos y evaluación."""
    if df is None or len(df) == 0:
        from src.modelos import CorpusInsuficienteError

        raise CorpusInsuficienteError(
            "No hay documentos cargados: carga un CSV o TXT con noticias."
        )
    if len(df) < n_temas:
        from src.modelos import CorpusInsuficienteError

        raise CorpusInsuficienteError(
            f"Corpus insuficiente: {len(df)} documentos para n_temas={n_temas}. "
            "Reduce el número de temas o agrega documentos."
        )

    textos = df["contenido"].fillna("").astype(str).tolist()
    textos_procesados = procesar_corpus(textos)

    # Ruta A: conteos para LDA. Ruta B: TF-IDF para NMF (mismo corpus,
    # vocabulario comparable por parámetros compartidos).
    vec_conteo, matriz_conteos = construir_matriz_conteos(textos_procesados)
    vec_tfidf, matriz_tfidf = construir_matriz_tfidf(textos_procesados)
    vocabulario = list(vec_conteo.get_feature_names_out())

    lda = entrenar_lda(matriz_conteos, n_temas=n_temas, semilla=semilla)
    nmf = entrenar_nmf(matriz_tfidf, n_temas=n_temas, semilla=semilla)

    resumenes = {
        "LDA": resumen_modelo(
            "LDA", lda, matriz_conteos, vocabulario, textos_procesados, n_palabras
        ),
        "NMF": resumen_modelo(
            "NMF", nmf, matriz_tfidf, vocabulario, textos_procesados, n_palabras
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
