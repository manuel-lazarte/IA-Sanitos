"""ETAPA 1 del agente: carga y validación de documentos - Grupo 15.

Acepta CSV con las columnas del informe (título, medio, fecha, URL y
contenido) y carpetas de archivos TXT (un documento por archivo).
Detecta documentos vacíos o demasiado cortos y duplicados (mismo
contenido o misma URL), informando cuántas filas se eliminaron.

Los encabezados se normalizan (sin tildes ni mayúsculas) para aceptar
archivos reales exportados con "Título", "URL", etc.
"""
from pathlib import Path
import unicodedata

import pandas as pd

COLUMNAS_REQUERIDAS = ("titulo", "medio", "fecha", "url", "contenido")
MIN_PALABRAS_CONTENIDO = 5


class DatosInvalidosError(ValueError):
    """El archivo de entrada no tiene la estructura mínima esperada."""


def _normalizar_encabezado(nombre: str) -> str:
    """'Título' -> 'titulo': minúsculas y sin tildes."""
    descompuesto = unicodedata.normalize("NFKD", str(nombre).strip().lower())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def _contenido_valido(texto: str, min_palabras: int = MIN_PALABRAS_CONTENIDO) -> bool:
    """Un documento válido tiene al menos `min_palabras` palabras."""
    if not isinstance(texto, str):
        return False
    return len(texto.split()) >= min_palabras


def _limpiar_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica filtros de calidad y devuelve (df_limpio, informe)."""
    filas_leidas = len(df)
    contenido = df["contenido"].fillna("").astype(str).str.strip()
    df = df.assign(contenido=contenido)

    mascara_valida = contenido.map(_contenido_valido)
    invalidos = int((~mascara_valida).sum())
    df = df[mascara_valida]

    # Duplicados: mismo contenido, o misma URL solo cuando la URL no está vacía
    # (los TXT cargados sin URL no deben eliminarse entre sí).
    tiene_url = df["url"].fillna("").str.strip().ne("")
    duplicada = df.duplicated(subset=["contenido"]) | (tiene_url & df.duplicated(subset=["url"]))
    duplicados = int(duplicada.sum())
    df = df[~duplicada]

    informe = {
        "filas_leidas": filas_leidas,
        "invalidos": invalidos,
        "duplicados": duplicados,
        "validos": len(df),
    }
    return df.reset_index(drop=True), informe


def cargar_csv(ruta: str | Path) -> tuple[pd.DataFrame, dict]:
    """Carga un CSV de noticias y devuelve (df_limpio, informe).

    Raise DatosInvalidosError si falta alguna columna requerida, nombrándola.
    """
    ruta = Path(ruta)
    df = pd.read_csv(ruta, encoding="utf-8")
    df.columns = [_normalizar_encabezado(c) for c in df.columns]

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise DatosInvalidosError(
            "Faltan columnas requeridas en el CSV: "
            + ", ".join(f"'{c}'" for c in faltantes)
            + ". Columnas requeridas: " + ", ".join(COLUMNAS_REQUERIDAS)
        )
    df = df[list(COLUMNAS_REQUERIDAS)].copy()
    return _limpiar_dataframe(df)


def cargar_carpeta_txt(ruta_carpeta: str | Path) -> tuple[pd.DataFrame, dict]:
    """Carga cada archivo *.txt de la carpeta como un documento.

    titulo = nombre del archivo, medio = 'Archivo TXT', fecha y url vacías.
    """
    carpeta = Path(ruta_carpeta)
    archivos = sorted(carpeta.glob("*.txt"))
    if not archivos:
        raise DatosInvalidosError(
            f"No se encontraron archivos .txt en la carpeta: {carpeta}"
        )
    filas = [
        {
            "titulo": a.stem,
            "medio": "Archivo TXT",
            "fecha": "",
            "url": "",
            "contenido": a.read_text(encoding="utf-8", errors="replace"),
        }
        for a in archivos
    ]
    df = pd.DataFrame(filas, columns=list(COLUMNAS_REQUERIDAS))
    return _limpiar_dataframe(df)
