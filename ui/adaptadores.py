"""Adaptador entre el pipeline y la interfaz - Grupo 15.

Normaliza la salida de cada método a una misma estructura y concentra los
filtros. No entrena nada ni toca `src/`: solo lee lo que ya produjo
`ejecutar_pipeline`.
"""
import unicodedata

import numpy as np
import pandas as pd

from src.modelos import normalizar_pesos

METODOS = ("LDA", "NMF", "BERTopic")

DESCRIPCIONES = {
    "LDA": "LDA trabaja sobre conteos de palabras: cada noticia es una mezcla de temas con proporciones.",
    "NMF": "NMF factoriza la matriz TF-IDF: sus pesos son coeficientes de contribución, no probabilidades.",
    "BERTopic": "BERTopic agruparía las noticias por similitud de significado y separaría las atípicas. "
                "Este método lo integra otro frente del grupo.",
}


def obtener_resultados_bertopic(pipeline: dict) -> dict | None:
    """PUNTO DE CONEXIÓN con BERTopic (otro frente). Hoy devuelve None.

    Cuando exista, debe devolver la misma estructura que `cargar_resultados`
    para LDA/NMF (más las noticias atípicas) y la UI activará la opción sola.
    """
    return None


def cargar_resultados(pipeline: dict, metodo: str) -> dict | None:
    """Resultados normalizados de un método, o None si aún no está disponible."""
    if metodo == "BERTopic":
        return obtener_resultados_bertopic(pipeline)
    if metodo == "LDA":
        vectorizador, matriz = pipeline["conteos"]
        modelo = pipeline["lda"]
    elif metodo == "NMF":
        vectorizador, matriz = pipeline["tfidf"]
        modelo = pipeline["nmf"]
    else:
        return None  # método desconocido: nunca se cuelgan datos de otro
    pesos = modelo.transform(matriz)
    return {
        "metodo": metodo,
        "resumen": pipeline["resumenes"][metodo],
        "pesos": pesos,
        "proporciones": normalizar_pesos(pesos),  # solo para presentar
        "vocabulario": list(vectorizador.get_feature_names_out()),
        "componentes": modelo.components_,
    }


def _sin_tildes(texto: str) -> str:
    """Minúsculas y sin acentos: «diesel» encuentra «diésel»."""
    normalizado = unicodedata.normalize("NFD", str(texto))
    return "".join(c for c in normalizado if not unicodedata.combining(c)).lower()


def filtrar_noticias(df: pd.DataFrame, medios=None, rango=None, texto: str = "") -> np.ndarray:
    """Posiciones (0-based) de las noticias que cumplen los filtros."""
    mascara = pd.Series(True, index=df.index)
    if medios:
        mascara &= df["medio"].isin(medios)
    if rango:
        fechas = pd.to_datetime(df["fecha"], errors="coerce")
        mascara &= fechas.between(pd.Timestamp(rango[0]), pd.Timestamp(rango[1]))
    if texto and texto.strip():
        cuerpo = (df["titulo"].fillna("") + " " + df["contenido"].fillna("")).map(_sin_tildes)
        mascara &= cuerpo.str.contains(_sin_tildes(texto), regex=False)
    return np.flatnonzero(mascara.to_numpy())


def conteo_por_tema(pesos: np.ndarray, indices: np.ndarray) -> list[int]:
    """Noticias (entre las filtradas) cuyo tema predominante es cada tema."""
    if len(indices) == 0:
        return [0] * pesos.shape[1]
    predominantes = pesos[indices].argmax(axis=1)
    return np.bincount(predominantes, minlength=pesos.shape[1]).tolist()


def noticias_representativas(res: dict, indices: np.ndarray, tema: int, n: int = 3):
    """[(posición, peso relativo)] de las noticias con más peso en el tema."""
    orden = indices[np.argsort(res["pesos"][indices, tema])[::-1]][:n]
    return [(int(i), float(res["proporciones"][i, tema])) for i in orden]


def pesos_palabras(res: dict, tema: int, n: int = 10) -> dict[str, float]:
    """Palabras principales del tema con su peso en el modelo."""
    comp = res["componentes"][tema]
    return {res["vocabulario"][int(i)]: float(comp[i]) for i in comp.argsort()[::-1][:n]}
