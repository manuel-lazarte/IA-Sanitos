"""ETAPA 1 del agente: carga y validación de documentos - Grupo 15.

Acepta CSV con las columnas del informe (título, medio, fecha, URL y
contenido) y carpetas de archivos TXT (un documento por archivo).
Detecta documentos vacíos o demasiado cortos y duplicados (mismo
contenido o misma URL), informando cuántas filas se eliminaron.

Robustez ante archivos reales (hallazgos de la batería de pruebas):
- Encabezados normalizados: acepta "Título", "URL", "Contenido ", etc.
- Codificación: intenta UTF-8 y recurre a Windows-1252 (Excel español).
- Delimitador: detecta coma, punto y coma o tabulador.
- Filas corruptas: si una fila tiene más/menos campos que el encabezado
  (p. ej. una coma sin citar), se rechaza el archivo nombrando la línea,
  porque pandas la desplazaría en silencio corrompiendo los datos.
"""
import csv
import io
import unicodedata
from pathlib import Path

import pandas as pd

COLUMNAS_REQUERIDAS = ("titulo", "medio", "fecha", "url", "contenido")
MIN_PALABRAS_CONTENIDO = 5
DELIMITADORES_CANDIDATOS = (",", ";", "\t")


class DatosInvalidosError(ValueError):
    """El archivo de entrada no tiene la estructura mínima esperada."""


def _normalizar_encabezado(nombre: str) -> str:
    """'Título' -> 'titulo': minúsculas y sin tildes."""
    descompuesto = unicodedata.normalize("NFKD", str(nombre).strip().lower())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def _palabras_con_letras(texto: str) -> int:
    """Número de tokens que contienen al menos una letra (',,,' no cuenta)."""
    return sum(1 for tok in (texto or "").split() if any(c.isalpha() for c in tok))


def _contenido_valido(texto: str, min_palabras: int = MIN_PALABRAS_CONTENIDO) -> bool:
    """Un documento válido tiene al menos `min_palabras` palabras con letras."""
    if not isinstance(texto, str):
        return False
    return _palabras_con_letras(texto) >= min_palabras


def _leer_texto_csv(ruta: Path) -> str:
    """Lee el archivo probando UTF-8 y luego Windows-1252 (Excel español).

    Devuelve el texto completo; si ninguna codificación funciona, error
    accionable en lugar de un UnicodeDecodeError en inglés.
    """
    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            return ruta.read_text(encoding=codificacion)
        except UnicodeDecodeError:
            continue
    raise DatosInvalidosError(
        f"El archivo '{ruta.name}' no es legible ni en UTF-8 ni en "
        "Windows-1252. Ábrelo en Excel o VS Code y guárdalo como 'CSV UTF-8'."
    )


def _detectar_delimitador(primera_linea: str) -> str:
    """Elige el delimitador que más columnas produce en el encabezado
    (coma, punto y coma o tabulador): los CSV de Excel español usan ';')."""
    mejor, mejor_n = DELIMITADORES_CANDIDATOS[0], 1
    for delimitador in DELIMITADORES_CANDIDATOS:
        n = primera_linea.count(delimitador) + 1
        if n > mejor_n:
            mejor, mejor_n = delimitador, n
    return mejor


def _texto_a_dataframe(texto: str, nombre: str) -> pd.DataFrame:
    """Valida la estructura cruda (delimitador, campos por fila) y devuelve
    el DataFrame con encabezados normalizados y columnas requeridas."""
    if not texto.strip():
        raise DatosInvalidosError(f"El archivo '{nombre}' está vacío.")

    primera = texto.splitlines()[0]
    delimitador = _detectar_delimitador(primera)
    filas = list(csv.reader(io.StringIO(texto), delimiter=delimitador))

    esperados = len(filas[0])
    corruptas = [
        (numero + 1, len(fila))
        for numero, fila in enumerate(filas)
        if any(campo.strip() for campo in fila) and len(fila) != esperados
    ]
    if corruptas:
        detalle = ", ".join(f"línea {n} tiene {c} campos (esperados {esperados})" for n, c in corruptas[:5])
        raise DatosInvalidosError(
            f"El archivo '{nombre}' tiene filas con el número de campos "
            f"incorrecto (suele ser una coma sin comillas en el texto): {detalle}. "
            "Corrige esas líneas o envuelve los textos con comillas."
        )
    if len(filas) < 2:
        raise DatosInvalidosError(
            f"El archivo '{nombre}' solo tiene encabezados, sin filas de datos."
        )

    df = pd.read_csv(io.StringIO(texto), sep=delimitador)
    df.columns = [_normalizar_encabezado(c) for c in df.columns]

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise DatosInvalidosError(
            "Faltan columnas requeridas en el CSV: "
            + ", ".join(f"'{c}'" for c in faltantes)
            + ". Columnas requeridas: " + ", ".join(COLUMNAS_REQUERIDAS)
        )
    return df[list(COLUMNAS_REQUERIDAS)].copy()


def _limpiar_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica filtros de calidad y devuelve (df_limpio, informe)."""
    if df.empty:
        raise DatosInvalidosError(
            "El CSV no contiene filas de datos (solo el encabezado)."
        )
    filas_leidas = len(df)
    contenido = df["contenido"].fillna("").astype(str).str.strip()
    df = df.assign(contenido=contenido)

    mascara_valida = contenido.map(_contenido_valido).astype(bool)
    invalidos = int((~mascara_valida).sum())
    df = df[mascara_valida]

    # Duplicados: mismo contenido (comparado sin espacios dobles, para
    # detectar republicaciones con espaciado distinto), o misma URL solo
    # cuando la URL no está vacía (los TXT cargados sin URL no se eliminan
    # entre sí). astype(str) evita el fallo de .str sobre columnas vacías.
    clave_contenido = df["contenido"].map(lambda t: " ".join(t.split()))
    tiene_url = df["url"].fillna("").astype(str).str.strip().ne("")
    duplicada = clave_contenido.duplicated() | (tiene_url & df["url"].duplicated())
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
    """Carga un CSV de noticias y devuelve (df_limpio, informe)."""
    ruta = Path(ruta)
    texto = _leer_texto_csv(ruta)
    df = _texto_a_dataframe(texto, ruta.name)
    return _limpiar_dataframe(df)


def cargar_carpeta_txt(ruta_carpeta: str | Path) -> tuple[pd.DataFrame, dict]:
    """Carga cada archivo .txt de la carpeta como un documento.

    titulo = nombre del archivo, medio = 'Archivo TXT', fecha y url vacías.
    Detecta extensiones en mayúsculas (.TXT) y distingue carpeta inexistente
    de carpeta sin archivos.
    """
    carpeta = Path(ruta_carpeta)
    if not carpeta.exists():
        raise DatosInvalidosError(f"La carpeta no existe: {carpeta}")
    if not carpeta.is_dir():
        raise DatosInvalidosError(f"La ruta no es una carpeta: {carpeta}")
    archivos = sorted(
        a for a in carpeta.iterdir() if a.suffix.lower() == ".txt" and a.is_file()
    )
    if not archivos:
        raise DatosInvalidosError(
            f"No se encontraron archivos .txt en la carpeta: {carpeta}"
        )
    filas = []
    for a in archivos:
        try:
            contenido = a.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # TXT guardado en cp1252/latin-1: decodificar sin sustitutos raros.
            contenido = a.read_text(encoding="latin-1")
        filas.append(
            {
                "titulo": a.stem,
                "medio": "Archivo TXT",
                "fecha": "",
                "url": "",
                "contenido": contenido,
            }
        )
    df = pd.DataFrame(filas, columns=list(COLUMNAS_REQUERIDAS))
    return _limpiar_dataframe(df)
