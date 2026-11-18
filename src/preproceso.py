"""ETAPA 2 del agente: preprocesamiento de textos en español - Grupo 15.

Pipeline por documento (informe 3.6):
  1. minúsculas y limpieza de caracteres innecesarios (URLs, signos, números)
  2. tokenización
  3. filtrado: palabras vacías del español (spaCy) + stopwords editoriales
     propias, tokens alfabéticos de 3+ caracteres
  4. lematización con el modelo es_core_news_sm de spaCy

Decisión clave del informe: NO se eliminan las tildes, porque fusionaría
palabras distintas («público» y «publicó»). La lematización preserva
acentos («exportaciones» -> «exportación»).
"""
from functools import lru_cache
import re

import spacy

NOMBRE_MODELO = "es_core_news_sm"
LONGITUD_MINIMA_TOKEN = 3

# Palabras frecuentes en noticias con escaso valor temático (editoriales).
# Se revisan a mano; quitar de aquí un término que aporte señal temática.
STOPWORDS_EXTRA = frozenset(
    {
        "según", "además", "también", "así", "aunque", "tras", "cada",
        "dice", "dijo", "añadió", "señaló", "manifestó", "expresó",
        "hoy", "ayer", "podría", "podrían", "será", "habrá",
        "mientras", "inclusive", "solamente", "respecto",
    }
)


@lru_cache(maxsize=1)
def obtener_nlp():
    """Carga (una sola vez) el modelo de español de spaCy."""
    try:
        nlp = spacy.load(NOMBRE_MODELO)
    except OSError as error:
        raise OSError(
            f"Falta el modelo de español '{NOMBRE_MODELO}'. Instálalo con: "
            f"python -m spacy download {NOMBRE_MODELO}"
        ) from error
    nlp.max_text_length = 2_000_000
    return nlp


def limpiar_texto(texto: str) -> str:
    """Minúsculas, sin URLs/emails/signos/números; solo letras latinas y espacios."""
    texto = (texto or "").lower()
    texto = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", texto)
    # Conserva a-z, tildes, ü, ñ y espacios; el resto se vuelve espacio.
    texto = re.sub(r"[^a-záéíóúüñ\s]+", " ", texto)
    return " ".join(texto.split())


def tokenizar(texto: str) -> list[str]:
    """Separa el texto limpio en palabras."""
    return limpiar_texto(texto).split()


def _token_aceptado(lemma: str, es_stop: bool) -> bool:
    """Filtro temático: alfabético, 3+ letras, fuera de stopwords."""
    if es_stop or lemma in STOPWORDS_EXTRA:
        return False
    return lemma.isalpha() and len(lemma) >= LONGITUD_MINIMA_TOKEN


def procesar_texto(texto: str) -> str:
    """Pipeline completo: devuelve los lemas útiles unidos por espacios."""
    nlp = obtener_nlp()
    tokens = tokenizar(texto)
    if not tokens:
        return ""
    doc = nlp(" ".join(tokens))
    lemas = [
        t.lemma_.lower()
        for t in doc
        if _token_aceptado(t.lemma_.lower(), t.is_stop)
    ]
    return " ".join(lemas)


def procesar_corpus(textos: list[str]) -> list[str]:
    """Procesa todos los documentos en lote (más rápido que doc por doc)."""
    nlp = obtener_nlp()
    limpios = [" ".join(tokenizar(t)) for t in textos]
    resultado: list[str] = []
    for doc in nlp.pipe(limpios, batch_size=32):
        lemas = [
            t.lemma_.lower()
            for t in doc
            if _token_aceptado(t.lemma_.lower(), t.is_stop)
        ]
        resultado.append(" ".join(lemas))
    return resultado


def es_documento_valido(texto: str, min_palabras: int = 5) -> bool:
    """Detecta documentos vacíos o demasiado cortos para aportar temas."""
    if not isinstance(texto, str):
        return False
    return len(texto.split()) >= min_palabras
