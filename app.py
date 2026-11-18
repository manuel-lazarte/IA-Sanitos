"""Explorador Inteligente de Temas - Grupo 15 (IA, UMSS).

Interfaz Streamlit del prototipo de agente de análisis documental por
descubrimiento de temas (aprendizaje no supervisado, LDA y NMF sobre
noticias en español). No es un chatbot: recibe documentos, ejecuta el
procesamiento y devuelve resultados interpretables de modelos reales.

Ejecutar:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.carga_datos import cargar_carpeta_txt, cargar_csv
from src.modelos import SEMILLA_ALEATORIA, indices_representativos
from src.pipeline import ejecutar_pipeline
from src.preproceso import procesar_corpus

ROOT = Path(__file__).resolve().parent
RUTA_DEMO = ROOT / "datos" / "noticias_demo.csv"
TITULO_APP = "Explorador Inteligente de Temas – Grupo 15"

# Colores fijos por modelo (color sigue a la entidad, nunca al rango).
COLOR_LDA = "#2a78d6"
COLOR_NMF = "#eb6834"

PAGINAS = [
    "Inicio",
    "1. Cargar noticias",
    "2. Preprocesamiento",
    "3. Modelo LDA",
    "4. Modelo NMF",
    "5. Comparación y evaluación",
]


# ------------------------------------------------------------------ helpers
def tabla_temas(resumen: dict) -> pd.DataFrame:
    """Resumen de un modelo -> tabla (Tema 1-based) para mostrar."""
    filas = [
        {
            "Tema": t["tema"] + 1,
            "Palabras": ", ".join(t["palabras"]),
            "Noticias predominantes": resumen["distribucion"].get(t["tema"], 0),
            "Coherencia": t["coherencia"],
        }
        for t in resumen["temas"]
    ]
    return pd.DataFrame(
        filas, columns=["Tema", "Palabras", "Noticias predominantes", "Coherencia"]
    )


def etiqueta_tema(id_tema: int, palabras: list[str], n: int = 5) -> str:
    """'Tema 3 · combustible, diésel, precio' (id interno 0-based -> 1-based)."""
    return f"Tema {id_tema + 1} · " + ", ".join(palabras[:n])


def grafico_barras(valores: dict[str, float], titulo: str, color: str, horizontal: bool = False):
    """Barras sencillas con ejes y grilla tenues (una sola serie, sin leyenda)."""
    if horizontal:
        fig, ax = plt.subplots(figsize=(6, max(2.0, 0.45 * len(valores))))
        etiquetas = list(valores.keys())[::-1]
        datos = list(valores.values())[::-1]
        ax.barh(etiquetas, datos, color=color, height=0.6)
        ax.tick_params(axis="y", labelsize=9)
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar([f"T{i + 1}" for i in range(len(valores))], list(valores.values()),
               color=color, width=0.6)
        ax.tick_params(axis="x", labelsize=9)
    ax.set_title(titulo, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=("x" if horizontal else "y"), color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _corpus_cargado() -> bool:
    return st.session_state.get("df") is not None


def _textos_procesados() -> bool:
    return st.session_state.get("textos_procesados") is not None


def _pipeline_ejecutado() -> bool:
    return st.session_state.get("pipeline") is not None


def _badge_datos_demo():
    if st.session_state.get("origen") == "demo":
        st.caption(
            "⚠️ Corpus DEMOSTRATIVO sintético: noticias ficticias generadas "
            "con medios inexistentes ('Medio Demo 1/2'). No proviene de "
            "medios reales."
        )


# ------------------------------------------------------------------ páginas
def pagina_inicio():
    st.header("Inicio")
    st.markdown(
        """
        ### ¿Qué es este sistema?
        Un **agente de análisis documental** basado en **aprendizaje no
        supervisado**: recibe noticias en español y **descubre temas
        recurrentes** sin usar categorías etiquetadas de antemano
        (informe, secciones 3.3 y 4.1).

        **Problema:** encontrar asuntos que se repiten en una colección de
        noticias bolivianas sin definir categorías previas como «política»
        o «economía».

        **Modelos comparados** (sobre el mismo corpus):
        - **LDA** – asignación latente de Dirichlet, sobre **conteos de palabras**.
        - **NMF** – factorización matricial no negativa, sobre **TF-IDF**.

        ### Cómo usarlo
        1. Carga noticias (CSV o TXT) o usa el corpus demo sintético.
        2. Procesa los textos (preprocesamiento en español).
        3. Entrena los modelos y explora los temas descubiertos.
        4. Compara LDA y NMF con diversidad y coherencia NPMI.

        Los **temas son interpretaciones posteriores**: el modelo entrega
        palabras y noticias; el nombre humano se propone leyendo esas
        salidas (informe, sección 3.9).
        """
    )
    st.info(
        "Este prototipo usa un corpus demo **sintético** de 30 noticias. "
        "El corpus real (200–400 noticias de medios bolivianos) se construirá "
        "en etapas posteriores según el informe del grupo.",
        icon="ℹ️",
    )


def pagina_cargar():
    st.header("1. Cargar noticias")
    st.caption("ETAPA 1 del agente: entrada de documentos.")
    fuente = st.radio(
        "Fuente de documentos",
        ["Corpus demo (sintético)", "Archivo CSV", "Archivos TXT"],
        horizontal=True,
    )

    if st.button("Cargar", type="primary"):
        with st.spinner("Cargando documentos…"):
            try:
                if fuente == "Corpus demo (sintético)":
                    df, informe = cargar_csv(RUTA_DEMO)
                    origen = "demo"
                elif fuente == "Archivo CSV":
                    subido = st.session_state.get("_csv_subido")
                    if subido is None:
                        st.warning("Primero selecciona un archivo CSV.")
                        return
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv", delete=False
                    ) as tmp:
                        tmp.write(subido.getvalue())
                        df, informe = cargar_csv(tmp.name)
                    origen = "csv"
                else:
                    subidos = st.session_state.get("_txt_subidos") or []
                    if not subidos:
                        st.warning("Primero selecciona uno o más archivos TXT.")
                        return
                    with tempfile.TemporaryDirectory() as carpeta:
                        for subido in subidos:
                            (Path(carpeta) / subido.name).write_bytes(
                                subido.getvalue()
                            )
                        df, informe = cargar_carpeta_txt(carpeta)
                    origen = "txt"
                st.session_state["df"] = df
                st.session_state["informe_carga"] = informe
                st.session_state["origen"] = origen
                # Reinicia etapas dependientes (corpus cambió).
                st.session_state.pop("textos_procesados", None)
                st.session_state.pop("pipeline", None)
                st.success(f"Cargados {informe['validos']} documentos válidos.")
            except Exception as error:  # errores de datos con mensaje claro
                st.error(f"No se pudo cargar: {error}")

    # Uploaders fuera del botón para que Streamlit los mantenga en re-runs.
    if fuente == "Archivo CSV":
        st.session_state["_csv_subido"] = st.file_uploader(
            "CSV con columnas: titulo, medio, fecha, url, contenido", type=["csv"]
        )
    elif fuente == "Archivos TXT":
        st.session_state["_txt_subidos"] = st.file_uploader(
            "Un archivo .txt por noticia", type=["txt"], accept_multiple_files=True
        )

    if _corpus_cargado():
        df = st.session_state["df"]
        informe = st.session_state["informe_carga"]
        st.subheader("Informe de carga")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Leídas", informe["filas_leidas"])
        c2.metric("Inválidas (cortas/vacías)", informe["invalidos"])
        c3.metric("Duplicadas", informe["duplicados"])
        c4.metric("Válidas", informe["validos"])
        _badge_datos_demo()
        st.subheader(f"Documentos ({len(df)})")
        vista = df.copy()
        vista["contenido"] = vista["contenido"].str.slice(0, 140) + "…"
        st.dataframe(vista, use_container_width=True, height=320)
    else:
        st.info("Aún no hay documentos cargados.")


def pagina_preproceso():
    st.header("2. Preprocesamiento")
    st.caption("ETAPA 2 del agente: limpieza y normalización de textos en español.")
    if not _corpus_cargado():
        st.info("Primero carga noticias en la página anterior.")
        return

    if st.button("Procesar textos", type="primary"):
        with st.spinner("Lematizando y filtrando con spaCy (es_core_news_sm)…"):
            df = st.session_state["df"]
            textos = df["contenido"].fillna("").astype(str).tolist()
            st.session_state["textos_procesados"] = procesar_corpus(textos)
            st.session_state.pop("pipeline", None)
        st.success("Textos procesados.")

    if not _textos_procesados():
        st.info(
            "El preprocesamiento convierte cada noticia en sus **lemas "
            "útiles**: minúsculas, sin URLs ni signos, sin palabras vacías "
            "del español, con tildes preservadas («público» ≠ «publicó»)."
        )
        return

    textos_procesados = st.session_state["textos_procesados"]
    df = st.session_state["df"]
    vacios = sum(1 for t in textos_procesados if not t.strip())
    c1, c2 = st.columns(2)
    c1.metric("Documentos procesados", len(textos_procesados))
    c2.metric("Quedaron sin términos útiles", vacios)

    st.subheader("Ejemplo: original vs. procesado")
    muestra = min(3, len(df))
    filas = [
        {
            "Original (recortado)": df.iloc[i]["contenido"][:220] + "…",
            "Procesado (lemas)": textos_procesados[i][:220] + "…",
        }
        for i in range(muestra)
    ]
    st.dataframe(pd.DataFrame(filas), use_container_width=True)
    _badge_datos_demo()


def _pagina_modelo(nombre_modelo: str, es_lda: bool):
    st.header(f"{'3.' if es_lda else '4.'} Modelo {nombre_modelo}")
    if es_lda:
        st.caption(
            "ETAPA 4 (MODELO 1): LDA sobre CONTEOS de palabras "
            "(CountVectorizer). Cada documento es una mezcla probabilística "
            "de temas (Blei et al., 2003)."
        )
    else:
        st.caption(
            "ETAPA 4 (MODELO 2): NMF sobre TF-IDF. Los pesos son "
            "coeficientes de la factorización, NO probabilidades como en "
            "LDA (informe 3.11)."
        )
    if not _textos_procesados():
        st.info("Primero carga noticias y procesa los textos (páginas 1 y 2).")
        return

    n_temas = st.slider("Número de temas", 2, 10, 6, key=f"n_temas_{nombre_modelo}")
    st.caption(f"Semilla aleatoria fija: {SEMILLA_ALEATORIA} (resultados reproducibles).")

    if st.button(f"Entrenar {nombre_modelo}", type="primary"):
        with st.spinner(f"Entrenando {nombre_modelo}…"):
            try:
                st.session_state["pipeline"] = ejecutar_pipeline(
                    st.session_state["df"], n_temas=n_temas
                )
            except Exception as error:
                st.error(f"No se pudo entrenar: {error}")
                return
        st.success("Análisis completo (LDA y NMF entrenados sobre el mismo corpus).")

    if not _pipeline_ejecutado():
        st.info("Entrena el modelo para ver los temas descubiertos.")
        return

    pipeline = st.session_state["pipeline"]
    resumen = pipeline["resumenes"][nombre_modelo]
    color = COLOR_LDA if es_lda else COLOR_NMF
    _badge_datos_demo()

    st.subheader("Temas descubiertos")
    st.dataframe(tabla_temas(resumen), use_container_width=True)

    # Pertenencia temática para noticias representativas.
    if es_lda:
        _, matriz = pipeline["conteos"]
        pesos = pipeline["lda"].transform(matriz)
    else:
        _, matriz = pipeline["tfidf"]
        pesos = pipeline["nmf"].transform(matriz)

    representantes = indices_representativos(pesos, n=3)
    df = pipeline["df"]

    for t in resumen["temas"]:
        with st.expander(etiqueta_tema(t["tema"], t["palabras"], n=10)):
            st.markdown(f"**Coherencia NPMI del tema:** {t['coherencia']:.3f}")
            if es_lda:
                vocab = list(pipeline["conteos"][0].get_feature_names_out())
                componentes = pipeline["lda"].components_[t["tema"]]
            else:
                vocab = list(pipeline["tfidf"][0].get_feature_names_out())
                componentes = pipeline["nmf"].components_[t["tema"]]
            pesos_palabras = {
                vocab[int(i)]: float(componentes[i])
                for i in componentes.argsort()[::-1][:10]
            }
            st.pyplot(
                grafico_barras(
                    pesos_palabras,
                    f"Peso de las palabras principales (tema {t['tema'] + 1})",
                    color,
                    horizontal=True,
                )
            )
            st.markdown("**Noticias representativas de este tema**")
            for idx in representantes[t["tema"]]:
                fila = df.iloc[idx]
                st.markdown(
                    f"- **{fila['titulo']}** — {fila['medio']} "
                    f"({fila['fecha']}) [enlace]({fila['url']})"
                )

    st.subheader("Distribución de documentos por tema predominante")
    distribucion = {t: float(n) for t, n in resumen["distribucion"].items()}
    st.pyplot(
        grafico_barras(distribucion, f"{nombre_modelo}: noticias por tema predominante", color)
    )
    st.caption(
        "Cada noticia puede tener peso en varios temas (mezcla); aquí se "
        "muestra solo su tema de mayor peso."
    )


def pagina_lda():
    _pagina_modelo("LDA", es_lda=True)


def pagina_nmf():
    _pagina_modelo("NMF", es_lda=False)


def pagina_comparacion():
    st.header("5. Comparación y evaluación")
    st.caption("ETAPA 6 del agente: evaluación inicial con medidas homogéneas (informe 3.13).")
    if not _pipeline_ejecutado():
        st.info("Entrena los modelos en las páginas 3 o 4 para comparar.")
        return
    pipeline = st.session_state["pipeline"]
    resumen_lda = pipeline["resumenes"]["LDA"]
    resumen_nmf = pipeline["resumenes"]["NMF"]
    _badge_datos_demo()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Diversidad LDA", f"{resumen_lda['diversidad']:.2f}")
    c2.metric("Diversidad NMF", f"{resumen_nmf['diversidad']:.2f}")
    c3.metric("Coherencia NPMI LDA", f"{resumen_lda['coherencia']:.3f}")
    c4.metric("Coherencia NPMI NMF", f"{resumen_nmf['coherencia']:.3f}")

    st.markdown(
        "**Diversidad** = proporción de términos únicos entre las listas de "
        "palabras principales. **Coherencia NPMI** = asociación por "
        "co-aparición de esas palabras, la misma medida y corpus para ambos "
        "modelos."
    )

    st.subheader("Temas lado a lado")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{resumen_lda['nombre']}**")
        st.dataframe(tabla_temas(resumen_lda), use_container_width=True)
    with col_b:
        st.markdown(f"**{resumen_nmf['nombre']}**")
        st.dataframe(tabla_temas(resumen_nmf), use_container_width=True)

    st.subheader("Distribución por tema predominante")
    col_c, col_d = st.columns(2)
    with col_c:
        st.pyplot(
            grafico_barras(
                {t: float(n) for t, n in resumen_lda["distribucion"].items()},
                "LDA", COLOR_LDA,
            )
        )
    with col_d:
        st.pyplot(
            grafico_barras(
                {t: float(n) for t, n in resumen_nmf["distribucion"].items()},
                "NMF", COLOR_NMF,
            )
        )

    st.markdown(
        "**Lectura de los resultados:** las métricas orientan, pero la "
        "pertinencia de cada tema se confirma leyendo las noticias "
        "representativas de las páginas 3 y 4 y proponiendo un nombre "
        "humano para cada tema (informe 3.13 y 4.5)."
    )


# ------------------------------------------------------------------ app
def main():
    st.set_page_config(page_title=TITULO_APP, page_icon="🔎", layout="wide")
    st.title(TITULO_APP)
    st.caption(
        "Aprendizaje no supervisado · PLN · Descubrimiento de temas en "
        "noticias bolivianas (prototipo académico)"
    )

    pagina = st.sidebar.radio("Navegación", PAGINAS)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Agente de análisis documental (no generativo). Prototipo del "
        "primer avance — Grupo 15, Gestión II-2026."
    )

    {
        "Inicio": pagina_inicio,
        "1. Cargar noticias": pagina_cargar,
        "2. Preprocesamiento": pagina_preproceso,
        "3. Modelo LDA": pagina_lda,
        "4. Modelo NMF": pagina_nmf,
        "5. Comparación y evaluación": pagina_comparacion,
    }[pagina]()


if __name__ == "__main__":
    main()
