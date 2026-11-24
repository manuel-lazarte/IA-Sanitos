"""Fragmentos HTML de la interfaz (funciones puras, sin Streamlit).

Todo texto que viene de los datos (títulos, medios, URLs) se escapa para
que un CSV no pueda inyectar HTML en la página.
"""
from html import escape

from ui.estilos import COLORES_METODO

PASOS = ("Cargar noticias", "Analizar", "Explorar temas")


def stepper_html(paso: int) -> str:
    partes = []
    for i, nombre in enumerate(PASOS, 1):
        clase = "done" if i < paso else "on" if i == paso else ""
        marca = "✓" if i < paso else str(i)
        partes.append(f'<span class="step {clase}"><i>{marca}</i>{nombre}</span>')
    return '<nav class="steps" aria-label="Pasos">' + "".join(partes) + "</nav>"


def chips_html(palabras: list[str]) -> str:
    return '<div class="chips">' + "".join(
        f'<span class="chip">{escape(p)}</span>' for p in palabras
    ) + "</div>"


def etiqueta_metodo_html(metodo: str) -> str:
    color = COLORES_METODO.get(metodo, "#1B1F24")
    return f'<span class="tag" style="background:{color}">{escape(metodo)}</span>'


def banner_demo_html() -> str:
    return (
        '<div class="banner"><b>Datos demostrativos.</b> Corpus sintético con '
        "noticias ficticias y medios inexistentes («Medio Demo»). No proviene "
        "de medios reales.</div>"
    )


def aviso_html(texto: str) -> str:
    return f'<div class="aviso">{escape(texto)}</div>'


def noticia_html(titulo, medio, fecha, url, extracto: str, peso: float | None = None) -> str:
    enlace = ""
    if str(url).startswith(("http://", "https://")):
        enlace = f' · <a href="{escape(str(url), quote=True)}" target="_blank" rel="noopener">Abrir enlace</a>'
    peso_txt = f" · Peso del tema en la noticia: {peso:.0%}" if peso is not None else ""
    return (
        f'<div class="noticia"><b>{escape(str(titulo))}</b>'
        f'<div class="meta">{escape(str(medio))} · {escape(str(fecha))}{enlace}{peso_txt}</div>'
        f"<p>{escape(extracto)}</p></div>"
    )
