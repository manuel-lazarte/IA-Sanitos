"""Explorador Inteligente de Temas - Grupo 15 (IA, UMSS).

Interfaz Streamlit del prototipo de agente de análisis documental por
descubrimiento de temas (aprendizaje no supervisado, LDA y NMF sobre
noticias en español). No es un chatbot: recibe documentos, ejecuta el
procesamiento y devuelve resultados interpretables de modelos reales.

La interfaz se organiza en 3 pasos (cargar → analizar → explorar) con un
stepper en el encabezado. El pipeline se entrena UNA sola vez y se guarda en
``st.session_state``: cambiar de paso, de tema o de filtro no reentrena.
La lógica de modelos vive en ``src/``; aquí solo se presenta. BERTopic queda
expuesto como botón sin funcionalidad (ver ``ui/adaptadores.py``).

Ejecutar:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.carga_datos import cargar_carpeta_txt, cargar_csv
from src.modelos import SEMILLA_ALEATORIA, CorpusInsuficienteError
from src.pipeline import ejecutar_pipeline
from ui import adaptadores, componentes, estilos

ROOT = Path(__file__).resolve().parent
RUTA_DEMO = ROOT / "datos" / "noticias_demo.csv"
TITULO_APP = "Explorador Inteligente de Temas – Grupo 15"

# Valor por defecto del número de temas (se puede cambiar en el paso 2 o en
# «Detalles metodológicos» del paso 3, siempre con re-analizar explícito).
N_TEMAS_DEFECTO = 6
#: Parámetro n_palabras que usa el pipeline por defecto (src/pipeline.py).
N_PALABRAS_PIPELINE = 10
#: Caracteres del extracto de cada noticia representativa.
EXTRACTO_MAX = 200

TODOS_LOS_MEDIOS = "Todos los medios"
TODAS_LAS_FECHAS = "Todas las fechas"


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


def etiquetas_temas(valores: dict) -> list[str]:
    """Etiquetas del eje T<n> con los IDs REALES de los temas (1-based).

    Nunca renumera por posición: si el tema 2 no tiene noticias
    predominantes, las claves {0: .., 2: ..} se etiquetan T1 y T3.
    """
    return [f"T{k + 1}" for k in valores]


def nombre_sugerido(palabras: list[str]) -> str:
    """Nombre derivado de las 3 primeras palabras (sugerencia, no categoría)."""
    return ", ".join(palabras[:3]) if palabras else "—"


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
        ax.bar(etiquetas_temas(valores), list(valores.values()),
               color=color, width=0.6)
        ax.tick_params(axis="x", labelsize=9)
    ax.set_title(titulo, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=("x" if horizontal else "y"), color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def mostrar_grafico(fig):
    """Muestra la figura y la CIERRA: sin esto, cada re-run de Streamlit
    acumula figuras en el gestor global de matplotlib (requisito del plan)."""
    st.pyplot(fig)
    plt.close(fig)


def _fmt(valor, decimales: int = 3) -> str:
    """Número con decimales fijos o '—' cuando no existe (sin inventar cifras)."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    return f"{float(valor):.{decimales}f}"


def _txt(valor) -> str:
    """Texto de una celda para mostrar: '—' cuando falta o es NaN."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    texto = str(valor).strip()
    return texto or "—"


def _extracto(valor) -> str:
    """Extracto recortado del contenido de una noticia."""
    texto = "" if valor is None or (isinstance(valor, float) and np.isnan(valor)) else str(valor)
    texto = texto.strip()
    if not texto:
        return "—"
    return texto[:EXTRACTO_MAX] + ("…" if len(texto) > EXTRACTO_MAX else "")


def tabla_comparativa(pipeline: dict) -> pd.DataFrame:
    """Tabla lado a lado LDA / NMF / BERTopic (lo que no existe, '—')."""
    resumenes = pipeline["resumenes"]
    bert = resumenes.get("BERTopic")
    config_bert = (bert or {}).get("configuracion")
    semilla_bert = (
        config_bert.get("semilla") if isinstance(config_bert, dict)
        else getattr(config_bert, "semilla", None)
    )
    lda, nmf = resumenes["LDA"], resumenes["NMF"]
    # Todas las celdas como texto: evita tipos mezclados (int/str) que
    # rompen la conversión a Arrow al mostrar la tabla.
    tabla = pd.DataFrame(
        {
            "LDA": [
                "Disponible",
                lda.get("n_temas"),
                _fmt(lda.get("coherencia")),
                _fmt(lda.get("diversidad"), 2),
                SEMILLA_ALEATORIA,
            ],
            "NMF": [
                "Disponible",
                nmf.get("n_temas"),
                _fmt(nmf.get("coherencia")),
                _fmt(nmf.get("diversidad"), 2),
                SEMILLA_ALEATORIA,
            ],
            "BERTopic": [
                "Disponible" if bert else "Aún no disponible",
                bert.get("n_temas") if bert else "—",
                # NPMI pendiente: sus descriptores llevan bigramas sin
                # lematizar y no son comparables con LDA/NMF (informe BERTopic).
                ("Pendiente*" if bert else "—"),
                _fmt(bert.get("diversidad"), 2) if bert else "—",
                str(semilla_bert if semilla_bert is not None else "—") if bert else "—",
            ],
        },
        index=["Estado", "Nº de temas", "Coherencia NPMI", "Diversidad", "Semilla"],
    )
    return tabla.astype(str)


def estado(titulo: str, mensaje: str, icono: str = "ℹ️") -> None:
    """Estado vacío o informativo con título y mensaje explicativo."""
    st.info(f"{icono} **{titulo}** — {mensaje}")


def _corpus_cargado() -> bool:
    return st.session_state.get("df") is not None


def _pipeline_ejecutado() -> bool:
    return st.session_state.get("pipeline") is not None


def _entrenar(df: pd.DataFrame, n_temas: int, incluir_bertopic: bool = False) -> bool:
    """Ejecuta el pipeline una vez y lo guarda en sesión. Devuelve True si salió bien."""
    with st.spinner(
        "Preprocesando textos y entrenando LDA y NMF"
        + (" y BERTopic (puede tardar: descarga embeddings)" if incluir_bertopic else "")
        + " (una sola vez)…"
    ):
        try:
            kwargs = {"modelos": ("LDA", "NMF", "BERTopic")} if incluir_bertopic else {}
            pipeline = ejecutar_pipeline(df, n_temas=int(n_temas), **kwargs)
        except ModuleNotFoundError as error:
            faltante = getattr(error, "name", None) or "un paquete opcional"
            st.error(
                "BERTopic necesita el paquete opcional que falta: "
                f"'{faltante}'. Instálalo con: pip install -r requirements-bertopic.txt "
                "(se recomienda Python 3.11-3.12; el análisis LDA/NMF sigue "
                "disponible sin BERTopic)."
            )
            return False
        except CorpusInsuficienteError as error:
            # El mensaje ya explica la causa y la solución (agregar docs o bajar temas).
            st.error(str(error))
            return False
        except Exception as error:
            st.error(
                f"No se pudo analizar: {error}. Revisa la carga de noticias "
                "del paso 1 o reduce el número de temas."
            )
            return False
    st.session_state["pipeline"] = pipeline
    st.session_state["tema_sel"] = 0
    return True


# ------------------------------------------------------------------ acciones
def ir_a(paso: int) -> None:
    """Callback de navegación: cambia de paso sin reentrenar nada."""
    st.session_state["paso"] = paso


def _elegir_metodo(nombre: str) -> None:
    st.session_state["metodo"] = nombre
    st.session_state["tema_sel"] = 0


def _seleccionar_tema(indice: int) -> None:
    st.session_state["tema_sel"] = indice


def _nuevo_analisis() -> None:
    """Reinicia el estado de la sesión (se ejecuta antes del rerun)."""
    claves_app = [
        "df", "informe_carga", "origen", "pipeline",
        "paso", "metodo", "tema_sel", "n_temas",
        "filtro_medio", "filtro_fecha", "filtro_texto",
        "_aviso_reanalizado",
    ]
    for clave in claves_app:
        st.session_state.pop(clave, None)
    st.session_state["paso"] = 1


# ------------------------------------------------------------------ pasos
def paso_cargar():
    st.header("Cargar noticias")
    st.caption("PASO 1: entrada de documentos al agente (CSV, TXT o corpus demo).")
    fuente = st.radio(
        "Fuente de documentos",
        ["Corpus demo (sintético)", "Archivo CSV", "Archivos TXT"],
        horizontal=True,
    )

    if st.button("Cargar", type="primary", key="btn_cargar"):
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
                    # Escribir y CERRAR el temporal antes de leerlo: sin flush,
                    # pandas abriría el archivo aún vacío (bug detectado en
                    # pruebas). Se borra siempre en el finally.
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv", delete=False
                    ) as tmp:
                        tmp.write(subido.getvalue())
                        ruta_tmp = tmp.name
                    try:
                        df, informe = cargar_csv(ruta_tmp)
                    finally:
                        Path(ruta_tmp).unlink(missing_ok=True)
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
                # El corpus cambió: se reinician etapas y filtros dependientes.
                for clave in ("pipeline", "tema_sel",
                              "filtro_medio", "filtro_fecha", "filtro_texto"):
                    st.session_state.pop(clave, None)
                st.success(f"Cargados {informe['validos']} documentos válidos.")
            except Exception as error:  # errores de datos con mensaje claro
                st.error(
                    f"No se pudo cargar: {error}. Revisa que el archivo tenga "
                    "las columnas titulo, medio, fecha, url y contenido."
                )

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
        if st.session_state.get("origen") == "demo":
            st.markdown(componentes.banner_demo_html(), unsafe_allow_html=True)
        st.subheader(f"Documentos ({len(df)})")
        vista = df.copy()
        vista["contenido"] = vista["contenido"].str.slice(0, 140) + "…"
        st.dataframe(vista, width="stretch", height=320)
        st.button(
            "Siguiente: Analizar →",
            key="btn_ir_analizar",
            on_click=ir_a,
            args=(2,),
            type="primary",
        )
    else:
        estado(
            "Sin corpus cargado",
            "Elige una fuente y pulsa «Cargar». El preprocesamiento forma "
            "parte del paso Analizar: no hay que procesar por separado.",
            icono="📄",
        )


def paso_analizar():
    st.header("Analizar")
    st.caption(
        "PASO 2: preprocesamiento + LDA + NMF + evaluación en un solo botón. "
        "El resultado se guarda y no se reentrena al cambiar de paso o filtro."
    )
    st.button("← Volver a cargar", key="btn_volver_1", on_click=ir_a, args=(1,))

    if not _corpus_cargado():
        estado(
            "Sin corpus cargado",
            "Primero carga noticias en el paso 1.",
            icono="📄",
        )
        return

    st.slider(
        "Número de temas",
        2,
        10,
        value=int(st.session_state.get("n_temas", N_TEMAS_DEFECTO)),
        key="n_temas",
        help="Valor por defecto 6. Puedes cambiarlo aquí o en "
             "«Detalles metodológicos» del paso 3.",
    )
    st.caption(
        f"Semilla aleatoria fija: {SEMILLA_ALEATORIA} (resultados reproducibles). "
        "El preprocesamiento (lematización en español) se ejecuta dentro de este paso."
    )

    st.checkbox(
        "Incluir BERTopic (ruta semántica opcional; requiere instalar "
        "`pip install -r requirements-bertopic.txt`, recomendado Python 3.11-3.12)",
        key="incluir_bertopic",
    )

    if st.button("Analizar noticias", type="primary", key="btn_analizar"):
        if _entrenar(st.session_state["df"], st.session_state.get("n_temas", N_TEMAS_DEFECTO),
                     incluir_bertopic=st.session_state.get("incluir_bertopic", False)):
            st.success("Análisis completo: LDA y NMF entrenados sobre el mismo corpus.")

    if not _pipeline_ejecutado():
        estado(
            "Aún sin análisis",
            "Pulsa «Analizar noticias» para procesar los textos y entrenar "
            "los dos modelos. No hay nada que tocar antes: el preprocesamiento "
            "está incluido.",
            icono="⚙️",
        )
        return

    pipeline = st.session_state["pipeline"]
    textos = pipeline.get("textos_procesados") or []
    vacios = sum(1 for t in textos if not str(t).strip())
    c1, c2 = st.columns(2)
    c1.metric("Documentos procesados", len(textos))
    c2.metric("Quedaron sin términos útiles", vacios)

    resumen_lda = pipeline["resumenes"]["LDA"]
    resumen_nmf = pipeline["resumenes"]["NMF"]
    c3, c4, c5, c6 = st.columns(4)
    c3.metric("Coherencia NPMI LDA", _fmt(resumen_lda.get("coherencia")))
    c4.metric("Coherencia NPMI NMF", _fmt(resumen_nmf.get("coherencia")))
    c5.metric("Diversidad LDA", _fmt(resumen_lda.get("diversidad"), 2))
    c6.metric("Diversidad NMF", _fmt(resumen_nmf.get("diversidad"), 2))

    if not resumen_lda.get("temas") and not resumen_nmf.get("temas"):
        estado(
            "El análisis no produjo resultados",
            "El corpus no dio temas utilizables. Agrega más noticias o "
            "baja el número de temas.",
            icono="⚠️",
        )
        return

    if st.session_state.get("origen") == "demo":
        st.markdown(componentes.banner_demo_html(), unsafe_allow_html=True)
    st.button(
        "Explorar temas →",
        key="btn_ir_explorar",
        on_click=ir_a,
        args=(3,),
        type="primary",
    )


def paso_explorar():
    st.header("Explorar temas")
    st.caption(
        "PASO 3: un tema por tarjeta. Toda tarjeta muestra el método de "
        "origen y qué es (o no es) el nombre del tema."
    )
    st.button("← Volver a analizar", key="btn_volver_2", on_click=ir_a, args=(2,))

    if not _pipeline_ejecutado():
        estado(
            "Aún sin análisis",
            "Ejecuta el análisis en el paso 2 para explorar los temas.",
            icono="🔍",
        )
        return

    pipeline = st.session_state["pipeline"]
    df = pipeline["df"]
    aviso = st.session_state.pop("_aviso_reanalizado", None)
    if aviso:
        st.success(f"Análisis rehecho con {aviso} temas. Ya ves los resultados nuevos.")
    if st.session_state.get("origen") == "demo":
        st.markdown(componentes.banner_demo_html(), unsafe_allow_html=True)

    # --- selector de método (botones; BERTopic sin funcionalidad) ---
    metodo = st.session_state.get("metodo", "LDA")
    if metodo not in adaptadores.METODOS:
        metodo = "LDA"
    cols = st.columns(3)
    for col, nombre in zip(cols, adaptadores.METODOS):
        activo = nombre == metodo
        with col:
            st.button(
                f"{'● ' if activo else ''}{nombre}",
                key=f"metodo_{nombre.lower()}",
                on_click=_elegir_metodo,
                args=(nombre,),
                width="stretch",
                type="primary" if activo else "secondary",
            )
    st.caption(adaptadores.DESCRIPCIONES[metodo])

    # --- filtros (solo acotan lo que se muestra) ---
    opciones_medio = [TODOS_LOS_MEDIOS] + sorted(df["medio"].dropna().unique().tolist())
    opciones_fecha = [TODAS_LAS_FECHAS] + sorted(
        df["fecha"].dropna().astype(str).unique().tolist(), reverse=True
    )
    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])
    medio = f1.selectbox("Medio", opciones_medio, key="filtro_medio")
    fecha = f2.selectbox("Fecha", opciones_fecha, key="filtro_fecha")
    texto = f3.text_input(
        "Búsqueda", key="filtro_texto", placeholder="palabra en título o contenido"
    )
    indices = adaptadores.filtrar_noticias(
        df,
        medios=None if medio == TODOS_LOS_MEDIOS else [medio],
        rango=None if fecha == TODAS_LAS_FECHAS else (fecha, fecha),
        texto=texto or "",
    )
    visibles = len(indices)
    st.caption(
        f"Mostrando {visibles} de {len(df)} noticias con los filtros actuales."
        if visibles < len(df)
        else f"Sin filtros activos: las {len(df)} noticias del corpus."
    )
    if visibles == 0:
        st.warning("Ninguna noticia coincide con los filtros actuales.")

    # Mismo modelo, otra vista: los filtros no reentrenan nada.
    res = adaptadores.cargar_resultados(pipeline, metodo)

    # --- columna de tarjetas + panel derecho ---
    izquierda, derecha = st.columns([1.1, 2], gap="large")
    with izquierda:
        if res is None:
            estado(
                "Aún no disponible",
                "BERTopic no está conectado al pipeline. Este método lo "
                "integra otro frente del grupo.",
                icono="🚧",
            )
        else:
            temas = res["resumen"].get("temas") or []
            if not temas:
                estado(
                    "Análisis sin resultados",
                    "El método no devolvió temas para este corpus.",
                    icono="⚠️",
                )
            else:
                conteos = adaptadores.conteo_por_tema(res["pesos"], indices)
                seleccion = st.session_state.get("tema_sel", 0)
                if seleccion >= len(temas):
                    seleccion = 0
                    st.session_state["tema_sel"] = 0
                for t in temas:
                    k = t["tema"]
                    activo = k == seleccion
                    fila_chips = " ".join(f"`{p}`" for p in t["palabras"][:4]) or "`—`"
                    etiqueta = (
                        f"{'✓ ' if activo else ''}"
                        f"**{nombre_sugerido(t['palabras'])}**\n\n"
                        f"{fila_chips}\n\n"
                        f"{conteos[k]} noticias · nombre sugerido"
                    )
                    st.button(
                        etiqueta,
                        key=f"tema_{k}",
                        on_click=_seleccionar_tema,
                        args=(k,),
                        width="stretch",
                        type="primary" if activo else "secondary",
                    )

    with derecha:
        if res is None:
            estado(
                "Aún no disponible",
                "BERTopic no está conectado al pipeline.",
                icono="🚧",
            )
            st.caption(
                "El botón se muestra desde ya para que la integración sea "
                "solo cambiar la fuente de datos (ver ui/adaptadores.py)."
            )
        else:
            temas = res["resumen"].get("temas") or []
            if not temas:
                estado(
                    "Análisis sin resultados",
                    "El método no devolvió temas para este corpus.",
                    icono="⚠️",
                )
            else:
                seleccion = st.session_state.get("tema_sel", 0)
                if seleccion >= len(temas):
                    seleccion = 0
                t = temas[seleccion]
                k = t["tema"]
                st.markdown(
                    componentes.etiqueta_metodo_html(metodo)
                    + f' <span class="nota">Tema {k + 1} de {len(temas)}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"#### {nombre_sugerido(t['palabras'])}")
                st.caption(
                    "Nombre sugerido, no es categoría del modelo. "
                    "Atribución por palabras y noticias, no por nombre."
                )

                st.markdown("**Palabras clave**")
                claves = list(adaptadores.pesos_palabras(res, k, n=10))
                st.markdown(componentes.chips_html(claves), unsafe_allow_html=True)

                st.markdown("**Noticias representativas**")
                reps = adaptadores.noticias_representativas(res, indices, k, n=3)
                if reps:
                    for pos, peso in reps:
                        fila = df.iloc[pos]
                        st.markdown(
                            componentes.noticia_html(
                                _txt(fila.get("titulo")),
                                _txt(fila.get("medio")),
                                _txt(fila.get("fecha")),
                                _txt(fila.get("url")),
                                _extracto(fila.get("contenido")),
                                peso,
                            ),
                            unsafe_allow_html=True,
                        )
                    st.caption("Se muestran las noticias más cercanas al centro del tema.")
                else:
                    st.info("Ninguna noticia coincide con los filtros actuales.")

                st.markdown(
                    componentes.aviso_html(
                        "Una noticia puede abordar varios asuntos. "
                        "Léela antes de aceptar el tema como correcto."
                    ),
                    unsafe_allow_html=True,
                )

    _detalles_metodologicos(pipeline, res)


def _detalles_metodologicos(pipeline: dict, res: dict | None) -> None:
    """Expander colapsado por defecto con métricas, comparación y parámetros."""
    with st.expander("Detalles metodológicos (para la defensa)", expanded=False):
        st.caption(
            "Las métricas se calculan con la misma fórmula y el mismo corpus "
            "para los tres métodos; esta pantalla solo las muestra."
        )
        if res is None:
            estado(
                "Sin métricas propias",
                "BERTopic aún no está conectado; su columna en la comparación "
                "queda en «Aún no disponible».",
                icono="🚧",
            )
        else:
            resumen = res["resumen"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Método", res["metodo"])
            c2.metric("Semilla", str(SEMILLA_ALEATORIA))
            c3.metric("Coherencia NPMI", _fmt(resumen.get("coherencia")))
            c4.metric("Diversidad", _fmt(resumen.get("diversidad"), 2))
            st.caption(
                "Coherencia NPMI = asociación por co-aparición de las palabras "
                "principales. Diversidad = proporción de términos únicos entre "
                "esas listas. Aquí se muestran, no se recalculan."
            )

        st.markdown("**Comparación lado a lado**")
        st.dataframe(tabla_comparativa(pipeline), width="stretch")

        if res is not None:
            resumen = res["resumen"]
            if resumen.get("temas"):
                st.markdown(f"**Temas del método activo ({res['metodo']})**")
                st.dataframe(tabla_temas(resumen), width="stretch")
                for t in resumen["temas"]:
                    st.caption(
                        f"{etiqueta_tema(t['tema'], t['palabras'], n=8)} — "
                        f"NPMI {_fmt(t.get('coherencia'))}"
                    )
                distribucion = {t: float(n) for t, n in resumen["distribucion"].items()}
                mostrar_grafico(
                    grafico_barras(
                        distribucion,
                        f"{res['metodo']}: noticias por tema predominante",
                        estilos.COLORES_METODO.get(res["metodo"], "#1B1F24"),
                    )
                )
                st.caption(
                    "Distribución sobre el corpus completo: cambiar los filtros "
                    "no reentrena el modelo. Cada noticia puede tener peso en "
                    "varios temas; aquí se muestra solo su tema de mayor peso."
                )

        st.markdown("**Parámetros**")
        p1, p2, p3 = st.columns(3)
        p1.metric("Semilla", SEMILLA_ALEATORIA)
        p2.metric("n_palabras (por tema)", N_PALABRAS_PIPELINE)
        n_temas = p3.slider(
            "n_temas",
            2,
            10,
            value=int(st.session_state.get("n_temas", N_TEMAS_DEFECTO)),
            key="n_temas",
        )
        entrenado = pipeline["resumenes"]["LDA"].get("n_temas")
        if int(n_temas) != int(entrenado or -1):
            st.warning(
                f"El análisis vigente usa {entrenado} temas y el control está "
                f"en {int(n_temas)}. Para aplicarlo hay que re-analizar "
                "(acción explícita, nunca automática)."
            )
            if st.button(
                f"Re-analizar con {int(n_temas)} temas",
                key="btn_reanalizar",
            ):
                if _entrenar(pipeline["df"], int(n_temas),
                             incluir_bertopic=st.session_state.get("incluir_bertopic", False)):
                    # Aviso diferido: st.rerun() borra lo que se pinte ahora.
                    st.session_state["_aviso_reanalizado"] = int(n_temas)
                    st.rerun()


# ------------------------------------------------------------------ app
def main():
    st.set_page_config(page_title=TITULO_APP, page_icon="🔎", layout="wide")
    estilos.inyectar_estilos()

    # Encabezado: título + stepper + botón de reinicio (reemplaza la nav lateral).
    c_titulo, c_pasos, c_nuevo = st.columns([2.4, 3.6, 1.2])
    with c_titulo:
        st.markdown(f"### {TITULO_APP}")
        st.caption(
            "Aprendizaje no supervisado · PLN · Descubrimiento de temas en "
            "noticias bolivianas"
        )
    with c_pasos:
        st.markdown(
            componentes.stepper_html(st.session_state.get("paso", 1)),
            unsafe_allow_html=True,
        )
    with c_nuevo:
        st.button(
            "Nuevo análisis",
            key="nuevo_analisis",
            on_click=_nuevo_analisis,
            width="stretch",
        )
    st.markdown("---")

    {
        1: paso_cargar,
        2: paso_analizar,
        3: paso_explorar,
    }.get(st.session_state.get("paso", 1), paso_cargar)()

    st.markdown("---")
    st.caption(
        "Agente de análisis documental (no generativo) · Prototipo académico · "
        "Grupo 15, UMSS. Los nombres de temas son sugerencias a partir de las "
        "palabras del modelo, no categorías fijas."
    )


if __name__ == "__main__":
    main()
