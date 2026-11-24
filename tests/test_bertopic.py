"""Unitarias sin descargas; integración real activada explícitamente."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.bertopic_modelo import (
    ConfigBERTopic, entrenar_bertopic, fragmentar_texto, resumir_bertopic,
)
from src.pipeline import ejecutar_pipeline


@pytest.fixture
def noticias():
    return pd.DataFrame({"contenido": [
        "Hospitales reciben vacunas para prevenir enfermedades en Bolivia.",
        "Médicos atienden pacientes en hospitales de Cochabamba.",
        "Futbolistas entrenan para disputar el torneo nacional.",
        "El equipo ganó el partido de fútbol del campeonato.",
        "Astrónomos descubren nuevas estrellas en la galaxia.",
    ]}, index=[9, 2, 2, 17, 31])


@pytest.mark.parametrize("asignaciones", [[0, 0, 1, 1, -1], [-1] * 5, [0] * 5])
def test_contrato_y_atipicos(noticias, asignaciones):
    modelo = SimpleNamespace(get_topic=lambda t: [("hospital", .4), ("", 0)],
                             get_representative_docs=lambda t: [noticias.iloc[0].contenido])
    r = resumir_bertopic(modelo, noticias, noticias.contenido.tolist(), asignaciones, {})
    assert r["n_temas"] == len(set(asignaciones) - {-1})
    assert sum(r["distribucion"].values()) == len(noticias)
    assert r["atipicos"]["cantidad"] == asignaciones.count(-1)
    assert r["atipicos"]["porcentaje"] == 100 * asignaciones.count(-1) / 5
    assert [d["indice_original"] for d in r["documentos"]] == [9, 2, 2, 17, 31]
    assert [d["doc_id"] for d in r["documentos"]] == list(range(5))
    assert all(t["tema"] != -1 and t["palabras"] == ["hospital"] for t in r["temas"])
    for t in r["temas"]:
        assert all(asignaciones[i] == t["tema"] for i in t["doc_ids"])
        assert set(t["representativos"]) <= set(t["doc_ids"])
    if not r["n_temas"]:
        assert r["diversidad"] is None and r["advertencias"]


@pytest.mark.parametrize("campo,valor", [("min_samples", 0), ("min_cluster_size", 1),
                                        ("n_neighbors", 1), ("batch_size", -1)])
def test_config_invalida(campo, valor):
    with pytest.raises(ValueError, match=campo):
        ConfigBERTopic(**{campo: valor})


@pytest.mark.parametrize("tipo", ["pequeno", "vacio", "nulo", "columna", "stopwords", "parametros"])
def test_entrada_problematicas(noticias, tipo):
    config = ConfigBERTopic()
    if tipo == "pequeno":
        noticias = noticias.iloc[:2]
    elif tipo == "vacio":
        noticias.iloc[0, 0] = " "
    elif tipo == "nulo":
        noticias.iloc[0, 0] = None
    elif tipo == "columna":
        noticias = noticias.rename(columns={"contenido": "otra"})
    elif tipo == "stopwords":
        noticias["contenido"] = "de la el y"
    else:
        config = ConfigBERTopic(min_samples=5)
    with pytest.raises(ValueError):
        entrenar_bertopic(noticias, config)


def test_fragmentos_preservan_final_y_limite():
    def tokenizer(texto, add_special_tokens, **kwargs):
        return {"input_ids": list(texto) + ([0, 0] if add_special_tokens else [])}
    texto = ("Noticia con tildes y negación. " * 70) + "FINAL_IMPORTANTE"
    fragmentos = fragmentar_texto(texto, tokenizer, 100)
    assert "".join(p for p, _ in fragmentos) == texto
    assert len(fragmentos) > 1
    assert all(len(tokenizer("query: " + p, True)["input_ids"]) <= 100 for p, _ in fragmentos)
    assert fragmentos[-1][0].endswith("FINAL_IMPORTANTE")


def test_ruta_semantica_no_ejecuta_lexicos(monkeypatch, noticias):
    import src.bertopic_modelo as semantico
    import src.pipeline as pipeline
    entrenar = Mock(return_value=(object(), {"nombre": "BERTopic"}))
    monkeypatch.setattr(semantico, "entrenar_bertopic", entrenar)
    prohibido = Mock(side_effect=AssertionError("No debe ejecutar ruta léxica"))
    monkeypatch.setattr(pipeline, "procesar_corpus", prohibido)
    r = ejecutar_pipeline(noticias, n_temas=100, modelos=("BERTopic",))
    assert set(r["resumenes"]) == {"BERTopic"}
    assert entrenar.call_args.args[0] is noticias
    prohibido.assert_not_called()


@pytest.mark.parametrize("modelos", [("LDA",), ("NMF",), ("LDA", "NMF")])
def test_rutas_lexicas_no_ejecutan_bertopic(monkeypatch, ruta_demo, modelos):
    import src.bertopic_modelo as semantico
    from src.carga_datos import cargar_csv
    prohibido = Mock(side_effect=AssertionError("BERTopic no solicitado"))
    monkeypatch.setattr(semantico, "entrenar_bertopic", prohibido)
    r = ejecutar_pipeline(cargar_csv(ruta_demo)[0], modelos=modelos)
    assert set(r["resumenes"]) == set(modelos)
    assert all(sum(s["distribucion"].values()) == 30 for s in r["resumenes"].values())
    prohibido.assert_not_called()


@pytest.mark.parametrize("modelos", [(), ("otro",), "BERTopic"])
def test_selector_invalido(noticias, modelos):
    with pytest.raises(ValueError, match="modelos"):
        ejecutar_pipeline(noticias, modelos=modelos)


def test_tres_rutas_conservan_resultados_lexicos(monkeypatch, ruta_demo):
    import src.bertopic_modelo as semantico
    from src.carga_datos import cargar_csv
    df, _ = cargar_csv(ruta_demo)
    base = ejecutar_pipeline(df)
    monkeypatch.setattr(semantico, "entrenar_bertopic",
                        Mock(return_value=(object(), {"nombre": "BERTopic"})))
    combinado = ejecutar_pipeline(df, modelos=("LDA", "NMF", "BERTopic"))
    for nombre in ("LDA", "NMF"):
        assert base["resumenes"][nombre] == combinado["resumenes"][nombre]
        np.testing.assert_array_equal(base[nombre.lower()].components_,
                                      combinado[nombre.lower()].components_)


@pytest.mark.skipif(os.getenv("RUN_BERTOPIC_INTEGRATION") != "1",
                    reason="Activar RUN_BERTOPIC_INTEGRATION=1; requiere pila opcional y E5")
def test_entrenamiento_real_demo(ruta_demo):
    from hdbscan import HDBSCAN
    from umap import UMAP
    from bertopic.vectorizers import ClassTfidfTransformer
    from src.carga_datos import cargar_csv

    df, _ = cargar_csv(ruta_demo)
    modelo, r = entrenar_bertopic(df, ConfigBERTopic(local_files_only=True))
    assert isinstance(modelo.hdbscan_model, HDBSCAN)
    assert isinstance(modelo.umap_model, UMAP)
    assert isinstance(modelo.ctfidf_model, ClassTfidfTransformer)
    assert modelo.umap_model.embedding_.shape[0] == len(df)
    assert modelo.hdbscan_model.labels_.shape == (len(df),)
    assert np.array_equal(np.array(r["asignaciones"]) == -1, modelo.hdbscan_model.labels_ == -1)
    assert r["n_documentos"] == 30 and sum(r["distribucion"].values()) == 30
    assert r["n_temas"] == len(set(modelo.topics_) - {-1})
    assert modelo.c_tf_idf_.shape[0] == len(set(modelo.topics_))
    vocabulario = set(modelo.vectorizer_model.get_feature_names_out())
    assert {"diésel", "sequía", "dólar", "fútbol"} <= vocabulario
    assert not {"disel", "sequa", "dlar", "ftbol"} & vocabulario
    assert all(t["palabras"] and t["doc_ids"] and t["representativos"] for t in r["temas"])
    assert len(r["configuracion"]["embeddings"]["fragmentos_por_documento"]) == 30
    # Fronteras verificadas también con el tokenizador E5 real, sin red.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    cfg = ConfigBERTopic(local_files_only=True)
    ruta = snapshot_download(cfg.modelo_embeddings, revision=cfg.revision,
                             cache_dir=cfg.cache_folder, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(ruta, local_files_only=True)
    largo = (df.iloc[0].contenido + " ") * 15 + "Última información relevante."
    partes = fragmentar_texto(largo, tokenizer, 512)
    assert len(partes) > 1 and "".join(p for p, _ in partes) == largo
    assert all(len(tokenizer("query: " + p)["input_ids"]) <= 512 for p, _ in partes)


@pytest.mark.skipif(os.getenv("RUN_BERTOPIC_INTEGRATION") != "1",
                    reason="Requiere pila opcional; embeddings controlados, clustering real")
def test_corpus_minimo_sin_grupos_real(monkeypatch, noticias):
    import src.bertopic_modelo as semantico
    # Solo se sustituye el codificador: UMAP/HDBSCAN/c-TF-IDF se ajustan realmente.
    monkeypatch.setattr(semantico, "generar_embeddings", lambda textos, config: (
        np.random.default_rng(42).normal(size=(len(textos), 12)).astype(np.float32), {}))
    modelo, r = entrenar_bertopic(noticias)
    assert len(modelo.hdbscan_model.labels_) == 5
    assert r["n_temas"] == 0  # 5 puntos no forman dos clusters de tamaño >=5
    assert r["atipicos"]["cantidad"] == 5 and r["atipicos"]["porcentaje"] == 100
    assert r["temas"] == [] and r["advertencias"]
