"""Prueba de integración end-to-end: CSV demo -> pipeline completo.

Demuestra que el sistema descubre temas REALES (salida de modelos
entrenados, no valores simulados) sobre el corpus sintético de 30
noticias con 6 temas intencionales, y que el resultado es determinista
(semilla fija: dos ejecuciones producen exactamente los mismos temas).
"""
import pytest

from src.carga_datos import cargar_csv
from src.modelos import CorpusInsuficienteError
from src.pipeline import ejecutar_pipeline

TEMAS_ESPERADOS = {
    "combustible", "diésel", "gasolina", "agua", "sequía", "precio",
    "dólar", "partido", "vacuna", "estudiante",
}


@pytest.fixture(scope="module")
def resultado_pipeline(ruta_demo):
    df, _ = cargar_csv(ruta_demo)
    return ejecutar_pipeline(df, n_temas=6)


def test_pipeline_produce_6_temas_por_modelo(resultado_pipeline):
    assert len(resultado_pipeline["resumenes"]) == 2
    for nombre in ("LDA", "NMF"):
        resumen = resultado_pipeline["resumenes"][nombre]
        assert resumen["n_temas"] == 6
        assert len(resumen["temas"]) == 6
        assert all(len(t["palabras"]) == 10 for t in resumen["temas"])


def test_los_temas_reflejan_el_corpus(resultado_pipeline):
    vocabulario_descubierto = {
        p
        for resumen in resultado_pipeline["resumenes"].values()
        for t in resumen["temas"]
        for p in t["palabras"]
    }
    encontradas = TEMAS_ESPERADOS & vocabulario_descubierto
    assert len(encontradas) >= 8, (
        f"Los modelos solo recuperaron {len(encontradas)} términos temáticos "
        f"esperados: {sorted(encontradas)}. Revisa el corpus demo o el "
        "preprocesamiento."
    )


def test_distribucion_cubre_todos_los_documentos(resultado_pipeline):
    for resumen in resultado_pipeline["resumenes"].values():
        assert sum(resumen["distribucion"].values()) == 30


def test_diversidad_alta_en_ambos_modelos(resultado_pipeline):
    for resumen in resultado_pipeline["resumenes"].values():
        assert resumen["diversidad"] > 0.5, (
            f"{resumen['nombre']} repite demasiado términos entre temas "
            f"(diversidad={resumen['diversidad']:.2f})"
        )


def test_pipeline_determinista(ruta_demo):
    """Dos ejecuciones con la misma semilla -> EXACTAMENTE los mismos temas.

    Prueba de que no hay resultados simulados: todo sale del entrenamiento.
    """
    df, _ = cargar_csv(ruta_demo)
    r1 = ejecutar_pipeline(df, n_temas=6)
    r2 = ejecutar_pipeline(df, n_temas=6)
    for nombre in ("LDA", "NMF"):
        palabras1 = [t["palabras"] for t in r1["resumenes"][nombre]["temas"]]
        palabras2 = [t["palabras"] for t in r2["resumenes"][nombre]["temas"]]
        assert palabras1 == palabras2


def test_pipeline_rechaza_corpus_mas_chico_que_los_temas(tmp_path):
    import pandas as pd

    df_chico = pd.DataFrame(
        {
            "titulo": ["A", "B"],
            "medio": ["Medio Demo 1"] * 2,
            "fecha": ["2026-11-01"] * 2,
            "url": ["https://demo.ejemplo/x/1", "https://demo.ejemplo/x/2"],
            "contenido": [
                "El diésel subió de precio en las estaciones del país entero",
                "La gasolina escasea y el transporte pide abastecimiento ya",
            ],
        }
    )
    with pytest.raises(CorpusInsuficienteError):
        ejecutar_pipeline(df_chico, n_temas=6)
