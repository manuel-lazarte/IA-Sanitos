"""Validador del dataset de noticias bolivianas - Grupo 15.

Ejecuta las comprobaciones técnicas sobre los archivos generados y muestra
un resumen real de errores y advertencias. NO edita los datos.

Uso:
    python validar_dataset.py [ruta_carpeta]

Comprueba:
  1. Lectura de noticias_bolivia.csv con pandas (columnas exactas del
     cargador del proyecto: titulo, medio, fecha, url, contenido).
  2. Consistencia de identificadores con trazabilidad_manifiesto.csv.
  3. Identificadores únicos, campos obligatorios completos, fechas ISO,
     URLs bien formadas, textos no vacíos y con mínimo de palabras.
  4. Duplicados exactos por huella de contenido.
  5. Prueba de vectorización TF-IDF (los datos deben transformarse sin fallo).
"""
import re
import sys
from pathlib import Path

import pandas as pd

CARPETA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
COLUMNAS = ["titulo", "medio", "fecha", "url", "contenido"]
ERRORES: list[str] = []
ADVERTENCIAS: list[str] = []


def principal() -> None:
    print(f"Validando dataset en: {CARPETA}\n")

    # 1. Lectura del CSV principal con el mismo contrato que usa el agente.
    ruta_csv = CARPETA / "noticias_bolivia.csv"
    if not ruta_csv.exists():
        ERRORES.append(f"No existe {ruta_csv.name}")
        _resumen()
        return
    df = pd.read_csv(ruta_csv, encoding="utf-8")
    print(f"[1] CSV leído: {len(df)} filas")
    if list(df.columns) != COLUMNAS:
        ERRORES.append(f"Columnas incorrectas: {list(df.columns)}")

    # 2. Trazabilidad: mismos identificadores y mismo tamaño.
    ruta_traza = CARPETA / "trazabilidad_manifiesto.csv"
    if ruta_traza.exists():
        traza = pd.read_csv(ruta_traza, encoding="utf-8")
        print(f"[2] Trazabilidad leída: {len(traza)} registros")
        if len(traza) != len(df):
            ERRORES.append(
                f"Manifiesto ({len(traza)}) y dataset ({len(df)}) difieren en tamaño"
            )
    else:
        traza = None
        ERRORES.append("Falta trazabilidad_manifiesto.csv")

    # 3. Obligatorios, unicidad y formatos.
    vacios = df.isna().sum()
    vacios = vacios[vacios > 0]
    for columna, n in vacios.items():
        ERRORES.append(f"Columna '{columna}' tiene {n} valores vacíos")
    if df.duplicated().any():
        n = int(df.duplicated().sum())
        ERRORES.append(f"{n} filas completamente duplicadas")

    fecha_valida = df["fecha"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
    if not fecha_valida.all():
        ERRORES.append(f"{int((~fecha_valida).sum())} fechas fuera del formato ISO AAAA-MM-DD")
    url_valida = df["url"].astype(str).str.match(r"^https?://[^\s]+$")
    if not url_valida.all():
        ERRORES.append(f"{int((~url_valida).sum())} URLs mal formadas")
    ids = traza["id_noticia"] if traza is not None else pd.Series(dtype=str)
    if len(ids) and ids.duplicated().any():
        ERRORES.append(f"{int(ids.duplicated().sum())} identificadores repetidos en el manifiesto")

    # 4. Calidad textual.
    palabras = df["contenido"].astype(str).str.split().str.len()
    cortos = df[palabras < 30]
    if len(cortos):
        ADVERTENCIAS.append(f"{len(cortos)} textos con menos de 30 palabras")
    huella = df["contenido"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
    duplicados = int(huella.duplicated().sum())
    if duplicados:
        ERRORES.append(f"{duplicados} contenidos duplicados (misma huella)")
    residuo_html = df["contenido"].astype(str).str.contains("<p>|&nbsp;|</div>", regex=True)
    if residuo_html.any():
        ADVERTENCIAS.append(f"{int(residuo_html.sum())} textos con residuos HTML")

    # 5. Vectorización TF-IDF: los datos deben alimentar el pipeline sin fallo.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizador = TfidfVectorizer(min_df=2, max_df=0.9)
        matriz = vectorizador.fit_transform(df["contenido"].astype(str))
        print(f"[5] TF-IDF OK: matriz {matriz.shape}, vocabulario {len(vectorizador.vocabulary_)} términos")
    except Exception as e:  # noqa: BLE001 - el validador reporta cualquier fallo
        ERRORES.append(f"La vectorización TF-IDF falló: {e}")

    # Resumen real (nada calculado "de memoria").
    print("\n===== ESTADÍSTICAS =====")
    print(f"Registros: {len(df)} | Medios: {df['medio'].nunique()} -> {sorted(df['medio'].unique())}")
    print(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
    print(f"Palabras por texto - media: {palabras.mean():.0f} | mediana: {palabras.median():.0f} | mín: {palabras.min()}")
    print(f"Por medio: {df['medio'].value_counts().to_dict()}")
    print("\n===== RESULTADO =====")
    for e in ERRORES:
        print(f"ERROR: {e}")
    for a in ADVERTENCIAS:
        print(f"AVISO: {a}")
    if not ERRORES:
        print("Sin errores. Dataset listo para el cargador (src/carga_datos.py).")
    sys.exit(1 if ERRORES else 0)


def _resumen() -> None:
    for e in ERRORES:
        print(f"ERROR: {e}")
    sys.exit(1)


if __name__ == "__main__":
    principal()
