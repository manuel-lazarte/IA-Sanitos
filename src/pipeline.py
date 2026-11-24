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

Por defecto conserva LDA + NMF. `modelos=("BERTopic",)` activa solo la
ruta semántica; las selecciones parciales devuelven solo sus artefactos.
`config_bertopic` permite ajustar ConfigBERTopic; cuando se proporciona,
sus valores prevalecen sobre semilla/n_palabras para esa ruta.
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


def _validar_corpus(df: pd.DataFrame, n_temas: int, requiere_temas: bool = True) -> None:
    """Falla pronto (antes de spaCy) si no hay documentos suficientes.

    La verificación de n_temas solo aplica a las rutas LDA/NMF: BERTopic
    (HDBSCAN) no fija el número de temas de antemano.
    """
    n_docs = 0 if df is None else len(df)
    if n_docs == 0:
        raise CorpusInsuficienteError(
            "No hay documentos cargados: carga un CSV o TXT con noticias."
        )
    if requiere_temas and n_docs < n_temas:
        raise CorpusInsuficienteError(
            f"Corpus insuficiente: {n_docs} documentos para n_temas={n_temas}. "
            "Reduce el número de temas o agrega documentos."
        )


def ejecutar_pipeline(
    df: pd.DataFrame,
    n_temas: int = 6,
    n_palabras: int = 10,
    semilla: int = SEMILLA_ALEATORIA,
    *,
    modelos: tuple[str, ...] = ("LDA", "NMF"),
    config_bertopic=None,
    textos_procesados: list[str] | None = None,
) -> dict:
    """Ejecuta preprocesamiento, representación, modelos y evaluación.

    Si la interfaz ya preprocesó los textos (página 2), puede pasarlos en
    `textos_procesados` para no repetir la lematización con spaCy.

    Por defecto conserva LDA + NMF. `modelos=("BERTopic",)` activa solo la
    ruta semántica; las selecciones parciales devuelven solo sus artefactos.
    `config_bertopic` permite ajustar ConfigBERTopic; cuando se proporciona,
    sus valores prevalecen sobre semilla/n_palabras para esa ruta.
    """
    if isinstance(modelos, str) or not modelos or set(modelos) - {"LDA", "NMF", "BERTopic"}:
        raise ValueError("modelos debe contener LDA, NMF y/o BERTopic.")
    _validar_corpus(df, n_temas, requiere_temas=bool(set(modelos) & {"LDA", "NMF"}))

    if set(modelos) != {"LDA", "NMF"}:
        resultado = {"df": df, "resumenes": {}}
        if set(modelos) & {"LDA", "NMF"}:
            # Preproceso perezoso: la ruta solo-semántica (BERTopic) no paga spaCy.
            if textos_procesados is None:
                textos_procesados = procesar_corpus(df["contenido"].fillna("").astype(str).tolist())
            resultado["textos_procesados"] = textos_procesados
            for nombre, clave, construir, entrenar in (
                ("LDA", "conteos", construir_matriz_conteos, entrenar_lda),
                ("NMF", "tfidf", construir_matriz_tfidf, entrenar_nmf),
            ):
                if nombre in modelos:
                    vec, matriz = construir(textos_procesados)
                    modelo = entrenar(matriz, n_temas=n_temas, semilla=semilla)
                    resultado[clave] = (vec, matriz)
                    resultado[nombre.lower()] = modelo
                    resultado["resumenes"][nombre] = resumen_modelo(
                        nombre, modelo, matriz, terminos_vocabulario(vec), textos_procesados, n_palabras
                    )
        if "BERTopic" in modelos:
            from src.bertopic_modelo import ConfigBERTopic, entrenar_bertopic

            config = config_bertopic or ConfigBERTopic(semilla=semilla, n_palabras=n_palabras)
            modelo, resumen = entrenar_bertopic(df, config)
            resultado["bertopic"] = modelo
            resultado["resumenes"]["BERTopic"] = resumen
        return resultado

    if textos_procesados is None:
        textos_procesados = procesar_corpus(df["contenido"].fillna("").astype(str).tolist())

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
