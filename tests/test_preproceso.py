"""Pruebas del preprocesamiento de textos en español (src/preproceso.py).

Punto crítico del informe (3.6): NO eliminar tildes («público»/«publicó»)
y lematizar preservando acentos («exportaciones» -> «exportación»).
"""
from src.preproceso import (
    es_documento_valido,
    limpiar_texto,
    procesar_corpus,
    procesar_texto,
    tokenizar,
)


def test_limpiar_texto_quita_urls_y_signos():
    sucio = "El PIB… visita https://x.com/noticia ¡Atención! al detalle."
    limpio = limpiar_texto(sucio)
    assert "https" not in limpio
    assert "!" not in limpio and "…" not in limpio
    assert limpio == limpio.lower()
    assert "atención" in limpio


def test_tokenizar_separa_palabras():
    assert tokenizar("Hola, mundo boliviano") == ["hola", "mundo", "boliviano"]


def test_procesar_texto_lematiza_y_quita_stopwords():
    procesado = procesar_texto("Las exportaciones subieron")
    assert "las" not in procesado.split()
    assert "subieron" not in procesado.split()
    assert "exportación" in procesado.split()


def test_procesar_texto_preserva_tildes_distintas():
    """«público» y «publicó» deben quedar como términos distintos."""
    publico = procesar_texto("El público asistió").split()
    publico_verb = procesar_texto("El museo publicó datos").split()
    assert "público" in publico
    assert "publicó" not in publico
    assert "publicar" in publico_verb
    assert publico_verb != publico


def test_procesar_texto_conserva_termino_con_tilde():
    assert "diésel" in procesar_texto("El diésel es caro").split()


def test_documento_vacio_o_corto_invalido():
    assert es_documento_valido("") is False
    assert es_documento_valido("Hola") is False
    assert es_documento_valido("   \n  ") is False


def test_noticia_demo_es_valida(ruta_demo):
    import pandas as pd

    df = pd.read_csv(ruta_demo)
    assert es_documento_valido(df.loc[0, "contenido"]) is True


def test_procesar_corpus_mismo_tamano():
    textos = ["El diésel subió de precio", "El equipo ganó el partido", ""]
    resultado = procesar_corpus(textos)
    assert len(resultado) == 3
    assert all(isinstance(t, str) for t in resultado)
