"""Ruta semántica opcional: E5 -> UMAP -> HDBSCAN -> c-TF-IDF.

No importa la pila pesada hasta solicitar entrenamiento. Los doc_ids son
posiciones iloc del DataFrame recibido, nunca etiquetas del índice pandas.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
import hashlib
import json
import platform

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.es.stop_words import STOP_WORDS

from src.evaluacion import diversidad_tematica
from src.preproceso import STOPWORDS_EXTRA


@dataclass(frozen=True)
class ConfigBERTopic:
    modelo_embeddings: str = "intfloat/multilingual-e5-small"
    revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    semilla: int = 42
    n_neighbors: int = 15
    n_components: int = 5
    min_cluster_size: int = 5
    min_samples: int = 2
    n_palabras: int = 10
    batch_size: int = 16
    cache_folder: str = ".cache/huggingface"
    local_files_only: bool = False

    def __post_init__(self):
        for campo, minimo in (("n_neighbors", 2), ("n_components", 2),
                              ("min_cluster_size", 2), ("min_samples", 1),
                              ("n_palabras", 1), ("batch_size", 1)):
            valor = getattr(self, campo)
            if not isinstance(valor, int) or isinstance(valor, bool) or valor < minimo:
                raise ValueError(f"{campo} debe ser un entero >= {minimo}.")


def fragmentar_texto(texto, tokenizer, limite):
    """Particiones contiguas, sin solapamiento ni pérdida del final.

    Comprueba cada entrada REAL con prefijo y tokens especiales antes de
    encode. La bisección solo afecta fronteras; no descarta caracteres.
    """
    if limite < 8:
        raise ValueError("El codificador necesita un límite de al menos 8 tokens.")
    pendientes = [texto]
    resultado = []
    while pendientes:
        parte = pendientes.pop()
        ids = tokenizer("query: " + parte, add_special_tokens=True,
                        truncation=False)["input_ids"]
        if len(ids) <= limite:
            resultado.append((parte, len(tokenizer(
                parte, add_special_tokens=False, truncation=False)["input_ids"])))
            continue
        mitad = len(parte) // 2
        corte = parte.rfind(" ", 0, mitad + 1)
        if corte <= 0:
            corte = mitad
        if corte <= 0:
            raise ValueError("No se puede fragmentar el texto con este codificador.")
        pendientes.extend([parte[corte:], parte[:corte]])
    return resultado


def generar_embeddings(textos, config):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download

    try:
        origen = config.modelo_embeddings
        if config.local_files_only:
            # Una ruta física evita que Transformers 4.57.6 consulte metadatos
            # del tokenizador en Hub pese a local_files_only=True.
            origen = snapshot_download(config.modelo_embeddings, revision=config.revision,
                                       cache_dir=config.cache_folder, local_files_only=True)
        encoder = SentenceTransformer(
            origen, revision=config.revision, device="cpu",
            cache_folder=config.cache_folder, local_files_only=config.local_files_only,
        )
    except OSError as exc:
        raise RuntimeError(
            "No se pudo cargar E5. La primera ejecución necesita descargar el modelo; "
            "revisa la conexión o la caché y local_files_only."
        ) from exc
    limite = min(512, encoder.max_seq_length)
    embeddings, cantidades = [], []
    for texto in textos:
        fragmentos = fragmentar_texto(texto, encoder.tokenizer, limite)
        vectores = encoder.encode(
            ["query: " + parte for parte, _ in fragmentos],
            batch_size=config.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        # Cada token aporta una vez; el fragmento final corto no pesa igual
        # que un fragmento largo. La media sacrifica orden entre fragmentos.
        vector = np.average(vectores, axis=0,
                            weights=[max(1, peso) for _, peso in fragmentos])
        norma = np.linalg.norm(vector)
        if not np.isfinite(norma) or norma == 0:
            raise ValueError("El codificador produjo un embedding vacío o no finito.")
        embeddings.append(vector / norma)
        cantidades.append(len(fragmentos))
    revision = getattr(encoder[0].auto_model.config, "_commit_hash", None)
    if config.local_files_only:
        from pathlib import Path
        revision = Path(origen).name
    return np.asarray(embeddings, dtype=np.float32), {
        "revision_resuelta": revision, "limite_tokens": limite,
        "fragmentos_por_documento": cantidades, "prefijo": "query: ",
        "agregacion": "media ponderada por tokens, normalizada L2, sin solapamiento",
        "device": "cpu",
    }


def resumir_bertopic(modelo, df, textos, asignaciones, configuracion):
    """Contrato de presentación; no interpreta fuerzas como mezclas LDA."""
    asignaciones = [int(t) for t in asignaciones]
    if len(asignaciones) != len(df):
        raise ValueError("BERTopic no devolvió una asignación por documento.")
    distribucion = dict(sorted(Counter(asignaciones).items()))
    temas = []
    for tema in sorted(set(asignaciones) - {-1}):
        ids = [i for i, t in enumerate(asignaciones) if t == tema]
        representantes = set(modelo.get_representative_docs(tema) or [])
        palabras = [(p, float(w)) for p, w in modelo.get_topic(tema)
                    if p and w > 0]
        temas.append({
            "tema": tema, "palabras": [p for p, _ in palabras],
            "pesos_ctfidf": [w for _, w in palabras], "doc_ids": ids,
            "representativos": [i for i in ids if textos[i] in representantes],
            "coherencia": None,
        })
    atipicos = [i for i, t in enumerate(asignaciones) if t == -1]
    return {
        "nombre": "BERTopic", "n_documentos": len(df), "n_temas": len(temas),
        "tema_ids": [t["tema"] for t in temas], "temas": temas,
        "asignaciones": asignaciones,
        "documentos": [{"doc_id": i, "indice_original": indice, "tema_id": tema}
                       for i, (indice, tema) in enumerate(zip(df.index, asignaciones))],
        "distribucion": distribucion,
        "atipicos": {"cantidad": len(atipicos), "proporcion": len(atipicos) / len(df),
                     "porcentaje": 100 * len(atipicos) / len(df), "doc_ids": atipicos},
        "diversidad": diversidad_tematica([t["palabras"] for t in temas]) if temas else None,
        "coherencia": None,
        "nota_evaluacion": "NPMI pendiente: los descriptores incluyen bigramas sin lematizar; "
                           "no son comparables directamente con los lemas de LDA/NMF.",
        "advertencias": (["No se encontraron grupos: revisa corpus y parámetros de densidad."]
                         if not temas else []),
        "configuracion": configuracion,
    }


def entrenar_bertopic(df, config=None):
    """Devuelve (BERTopic ajustado, resumen). No filtra ni reordena filas."""
    config = config or ConfigBERTopic()
    if df is None or len(df) < 5:
        raise ValueError("BERTopic necesita al menos 5 noticias; se recomiendan 200–400.")
    if "contenido" not in df:
        raise ValueError("Falta la columna contenido para BERTopic.")
    textos = df["contenido"].tolist()
    if any(not isinstance(t, str) or not t.strip() for t in textos):
        raise ValueError("Cada noticia debe tener contenido textual no vacío; revisa la carga.")
    textos = [" ".join(t.split()) for t in textos]
    if config.min_cluster_size > len(textos) or config.min_samples >= len(textos):
        raise ValueError("Reduce min_cluster_size/min_samples o agrega noticias al corpus.")
    vectorizer = CountVectorizer(
        stop_words=sorted(STOP_WORDS | STOPWORDS_EXTRA), ngram_range=(1, 2),
        min_df=1, token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
    )
    if not any(vectorizer.build_analyzer()(t) for t in textos):
        raise ValueError("El corpus no contiene vocabulario útil en español para c-TF-IDF.")
    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("Instala la ruta opcional: python -m pip install -r "
                          "requirements-bertopic.txt") from exc
    embeddings, info_embeddings = generar_embeddings(textos, config)
    umap = UMAP(n_neighbors=min(config.n_neighbors, len(textos) - 1),
                n_components=min(config.n_components, len(textos) - 2),
                min_dist=0.0, metric="cosine", random_state=config.semilla,
                n_jobs=1, init="random")
    hdbscan = HDBSCAN(min_cluster_size=config.min_cluster_size,
                      min_samples=config.min_samples, metric="euclidean",
                      cluster_selection_method="eom", prediction_data=True,
                      core_dist_n_jobs=1)
    modelo = BERTopic(language="multilingual", embedding_model=None,
                      umap_model=umap, hdbscan_model=hdbscan,
                      vectorizer_model=vectorizer, ctfidf_model=ClassTfidfTransformer(),
                      top_n_words=config.n_palabras, calculate_probabilities=False,
                      nr_topics=None)
    asignaciones, _ = modelo.fit_transform(textos, embeddings=embeddings)
    configuracion = {
        "solicitada": asdict(config), "embeddings": info_embeddings,
        "umap": umap.get_params(), "hdbscan": hdbscan.get_params(),
        "vectorizador": {"ngram_range": [1, 2], "min_df": 1,
                         "stop_words": vectorizer.stop_words,
                         "token_pattern": vectorizer.token_pattern},
        "ctfidf": modelo.ctfidf_model.get_params(),
        "versiones": {p: version(p) for p in (
            "bertopic", "sentence-transformers", "transformers", "torch",
            "umap-learn", "hdbscan", "numpy", "scikit-learn", "numba", "spacy")},
        "python": platform.python_version(), "plataforma": platform.platform(),
        "corpus_sha256": hashlib.sha256(json.dumps(textos, ensure_ascii=False).encode()).hexdigest(),
    }
    # Parámetros como Memory e infinito no son valores JSON interoperables.
    configuracion = json.loads(json.dumps(configuracion, default=str), parse_constant=str)
    return modelo, resumir_bertopic(modelo, df, textos, asignaciones, configuracion)
