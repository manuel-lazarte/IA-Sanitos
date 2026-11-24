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
    "BERTopic": "BERTopic agrupa las noticias por similitud de significado (embeddings) y separa las "
                "noticias atípicas. Es la ruta opcional: activa «Incluir BERTopic» al analizar "
                "(requiere el paquete de requirements-bertopic.txt).",
}


def obtener_resultados_bertopic(pipeline: dict) -> dict | None:
    """PUNTO DE CONEXIÓN con BERTopic: mismo contrato que LDA/NMF.

    Devuelve None si el pipeline no incluye BERTopic. Las diferencias del
    método se respetan (no se inventan cifras):
    - pesos: matriz binaria doc-tema indexada por el ID CRUDO de tema de
      BERTopic; los atípicos (-1) no suman a ningún tema.
    - noticias representativas: las que el propio modelo señala, sin peso
      relativo (ver noticias_representativas).
    - palabras: pesos c-TF-IDF reales por tema (ver pesos_palabras).
    """
    resumen = pipeline.get("resumenes", {}).get("BERTopic")
    if resumen is None:
        return None
    n_docs = resumen.get("n_documentos") or 0
    temas = resumen.get("temas") or []
    asignaciones = resumen.get("asignaciones") or []
    n_columnas = max([t["tema"] for t in temas] + [0]) + 1 if temas else 1
    pesos = np.zeros((n_docs, n_columnas))
    for doc, tema in enumerate(asignaciones):
        if 0 <= tema < n_columnas:
            pesos[doc, tema] = 1.0
    return {
        "metodo": "BERTopic",
        "resumen": resumen,
        "pesos": pesos,
        "proporciones": pesos,  # asignación binaria, no probabilidades
        "vocabulario": None,
        "componentes": None,
        "temas": temas,
        "asignaciones": asignaciones,
        "representativos_por_tema": {t["tema"]: t["representativos"] for t in temas},
    }


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


def conteo_por_tema(
    pesos: np.ndarray, indices: np.ndarray, asignaciones: list[int] | None = None
) -> list[int]:
    """Noticias (entre las filtradas) cuyo tema predominante es cada tema.

    Con `asignaciones` (BERTopic) se cuenta por ID crudo de tema y los
    atípicos (-1) no se suman a ningún tema.
    """
    if len(indices) == 0:
        return [0] * pesos.shape[1]
    if asignaciones is not None:
        return [sum(asignaciones[i] == k for i in indices) for k in range(pesos.shape[1])]
    predominantes = pesos[indices].argmax(axis=1)
    return np.bincount(predominantes, minlength=pesos.shape[1]).tolist()


def noticias_representativas(res: dict, indices: np.ndarray, tema: int, n: int = 3):
    """[(posición, peso relativo)] de las noticias con más peso en el tema.

    Para BERTopic: las que el modelo señala como representativas (sin peso
    relativo — None —, porque HDBSCAN no produce pesos por noticia).
    """
    representativos = res.get("representativos_por_tema")
    if representativos is not None:
        dentro = set(indices.tolist())
        elegidos = [i for i in representativos.get(tema, []) if i in dentro][:n]
        return [(int(i), None) for i in elegidos]
    orden = indices[np.argsort(res["pesos"][indices, tema])[::-1]][:n]
    return [(int(i), float(res["proporciones"][i, tema])) for i in orden]


def pesos_palabras(res: dict, tema: int, n: int = 10) -> dict[str, float]:
    """Palabras principales del tema con su peso en el modelo.

    Para BERTopic: los pesos c-TF-IDF reales que guardó el resumen.
    """
    temas = res.get("temas")
    if temas is not None:
        tema_datos = next((t for t in temas if t["tema"] == tema), None)
        if tema_datos is None:
            return {}
        pares = list(zip(tema_datos["palabras"], tema_datos["pesos_ctfidf"]))[:n]
        return {p: float(w) for p, w in pares}
    comp = res["componentes"][tema]
    return {res["vocabulario"][int(i)]: float(comp[i]) for i in comp.argsort()[::-1][:n]}
