"""Pruebas de carga y validación de documentos (src/carga_datos.py)."""
from pathlib import Path

import pandas as pd
import pytest

from src.carga_datos import DatosInvalidosError, cargar_carpeta_txt, cargar_csv


def test_carga_csv_demo_completa(ruta_demo):
    df, informe = cargar_csv(ruta_demo)
    assert len(df) == 30
    assert list(df.columns) == ["titulo", "medio", "fecha", "url", "contenido"]
    assert informe["validos"] == 30
    assert informe["duplicados"] == 0
    assert informe["invalidos"] == 0


def test_carga_normaliza_encabezados_con_tildes(tmp_path):
    ruta = tmp_path / "noticia.csv"
    ruta.write_text(
        "Título,Medio,Fecha,URL,Contenido\n"
        "Nota de prueba,Medio Demo 1,2026-11-01,https://demo.ejemplo/x/001,"
        "Texto con suficientes palabras para ser un documento valido aqui.\n",
        encoding="utf-8",
    )
    df, _ = cargar_csv(ruta)
    assert len(df) == 1
    assert df.loc[0, "titulo"] == "Nota de prueba"


def test_columna_faltante_lanza_error_con_nombre(tmp_path):
    ruta = tmp_path / "incompleto.csv"
    ruta.write_text("titulo,medio,fecha,url\nA,Medio,2026-01-01,https://x\n", encoding="utf-8")
    with pytest.raises(DatosInvalidosError) as exc:
        cargar_csv(ruta)
    assert "contenido" in str(exc.value)


def test_invalidos_y_duplicados_contados(tmp_path):
    largo = "un texto largo con muchas palabras para pasar el minimo establecido"
    ruta = tmp_path / "sucio.csv"
    ruta.write_text(
        "titulo,medio,fecha,url,contenido\n"
        f"N1,Medio Demo 1,2026-11-01,https://demo.ejemplo/a/001,{largo}\n"
        "Corta,Medio Demo 1,2026-11-02,https://demo.ejemplo/a/002,muy poco\n"
        f"Duplicada,Medio Demo 2,2026-11-03,https://demo.ejemplo/a/003,{largo}\n",
        encoding="utf-8",
    )
    df, informe = cargar_csv(ruta)
    assert len(df) == 1
    assert informe["invalidos"] == 1
    assert informe["duplicados"] == 1
    assert informe["validos"] == 1


def test_carga_carpeta_txt(tmp_path):
    for n in range(3):
        (tmp_path / f"nota_{n}.txt").write_text(
            f"Cuerpo de la noticia numero {n} con bastantes palabras para ser valida.",
            encoding="utf-8",
        )
    df, informe = cargar_carpeta_txt(tmp_path)
    assert len(df) == 3
    assert set(df["medio"]) == {"Archivo TXT"}
    assert informe["validos"] == 3


# --- Hallazgos de la batería de 5 agentes (correcciones) ---

def test_fila_con_campo_de_mas_lanza_error_nombrando_linea(tmp_path):
    """CRÍTICO: coma sin citar => fila desplazada en silencio. Debe rechazarse
    nombrando la línea, no cargarse como válida."""
    ruta = tmp_path / "corrupto.csv"
    ruta.write_text(
        "titulo,medio,fecha,url,contenido\n"
        "t1,M,2026-11-01,u1,texto con coma, este texto final tiene seis palabras\n",
        encoding="utf-8",
    )
    with pytest.raises(DatosInvalidosError) as exc:
        cargar_csv(ruta)
    assert "2" in str(exc.value)  # la línea 2 (la primera de datos)


def test_csv_solo_encabezados_lanza_error_claro(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_text("titulo,medio,fecha,url,contenido\n", encoding="utf-8")
    with pytest.raises(DatosInvalidosError) as exc:
        cargar_csv(ruta)
    assert "sin filas" in str(exc.value).lower()


def test_csv_sin_filas_validas_no_crashea(tmp_path):
    """Todas inválidas + columna url toda vacía (dtype float en pandas):
    debe devolver informe validos=0, no un AttributeError."""
    ruta = tmp_path / "nada_valido.csv"
    ruta.write_text(
        "titulo,medio,fecha,url,contenido\n"
        "t1,M,2026-11-01,,corta\n"
        "t2,M,2026-11-02,,corta\n",
        encoding="utf-8",
    )
    df, informe = cargar_csv(ruta)
    assert len(df) == 0
    assert informe["validos"] == 0


def test_csv_cp1252_se_carga_con_tildes(tmp_path):
    """CSV guardado por Excel en español (cp1252): debe cargar con tildes."""
    ruta = tmp_path / "excel.csv"
    ruta.write_bytes(
        "titulo,medio,fecha,url,contenido\n"
        "María leyó,Medio,2026-11-01,u1,"
        "texto con tilde María participación año suficientemente largo\n".encode("cp1252")
    )
    df, _ = cargar_csv(ruta)
    assert len(df) == 1
    assert "María" in df.loc[0, "titulo"]


def test_csv_separador_punto_y_coma_se_detecta(tmp_path):
    ruta = tmp_path / "excel_es.csv"
    ruta.write_text(
        "titulo;medio;fecha;url;contenido\n"
        "N1;Medio Demo 1;2026-11-01;u1;texto con suficiente cantidad de palabras aqui\n",
        encoding="utf-8",
    )
    df, _ = cargar_csv(ruta)
    assert len(df) == 1
    assert df.loc[0, "titulo"] == "N1"


def test_duplicados_con_espacios_internos_distintos(tmp_path):
    largo = "uno dos tres cuatro cinco seis"
    ruta = tmp_path / "dup_espacios.csv"
    ruta.write_text(
        "titulo,medio,fecha,url,contenido\n"
        f"A,M,2026-11-01,u1,{largo}\n"
        f"B,M,2026-11-02,u2,uno  dos  tres  cuatro  cinco  seis\n",
        encoding="utf-8",
    )
    _, informe = cargar_csv(ruta)
    assert informe["duplicados"] == 1


def test_token_solo_puntuacion_no_cuenta_como_palabra(tmp_path):
    ruta = tmp_path / "puntuacion.csv"
    ruta.write_text(
        "titulo,medio,fecha,url,contenido\n"
        't1,M,2026-11-01,u1,"uno dos tres cuatro ,,,"\n',
        encoding="utf-8",
    )
    _, informe = cargar_csv(ruta)
    assert informe["invalidos"] == 1  # solo 4 palabras reales


def test_carpeta_txt_mayusculas_detectadas(tmp_path):
    (tmp_path / "NOTICIA.TXT").write_text(
        "Texto con suficientes palabras para ser valido aqui dentro.", encoding="utf-8"
    )
    df, _ = cargar_carpeta_txt(tmp_path)
    assert len(df) == 1


def test_carpeta_inexistente_mensaje_diferente(tmp_path):
    with pytest.raises(DatosInvalidosError) as exc:
        cargar_carpeta_txt(tmp_path / "no_existe_123")
    assert "no existe" in str(exc.value).lower()


def test_txt_cp1252_sin_caracteres_de_reemplazo(tmp_path):
    (tmp_path / "nota.txt").write_bytes(
        "La calificación de María en España fue básicamente un éxito.".encode("cp1252")
    )
    df, _ = cargar_carpeta_txt(tmp_path)
    assert "calificación" in df.loc[0, "contenido"]
    assert "�" not in df.loc[0, "contenido"]
