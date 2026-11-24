"""Estilos de la interfaz (CSS inyectado en Streamlit) - Grupo 15."""
import streamlit as st

COLORES_METODO = {"LDA": "#1F5FB0", "NMF": "#B8461A", "BERTopic": "#0F6B5F"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: Inter, system-ui, sans-serif; }
.stApp { background: #F6F6F2; color: #1B1F24; }
.block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
.brand b { font-size: 1.15rem; display: block; }
.brand span, .nota { font-size: .85rem; color: #4A515B; }
.steps { display: flex; gap: 8px; flex-wrap: wrap; }
.step { display: flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px;
        font-size: .9rem; color: #4A515B; background: #ECECE5; }
.step i { font-style: normal; width: 22px; height: 22px; border-radius: 50%; background: #C9CBC3;
          color: #1B1F24; display: inline-flex; align-items: center; justify-content: center;
          font-size: .75rem; font-weight: 700; }
.step.on { background: #1B1F24; color: #fff; }
.step.on i { background: #fff; }
.step.done i { background: #0F6B5F; color: #fff; }
.banner { background: #FFF4D6; border: 1px solid #E8C766; border-radius: 12px;
          padding: 12px 16px; font-size: .9rem; color: #5C4300; margin: 8px 0 16px; }
.aviso { background: #FFF4D6; border-radius: 10px; padding: 10px 14px; font-size: .85rem;
         color: #5C4300; margin-top: 12px; }
.lead { font-size: 1.05rem; line-height: 1.55; color: #363C45; max-width: 860px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.chip { font-size: .82rem; padding: 4px 10px; border-radius: 999px; background: #EDEDE6; color: #2B3038; }
.tag { display: inline-block; font-size: .75rem; font-weight: 700; padding: 3px 10px;
       border-radius: 999px; color: #fff; margin-right: 8px; }
.tema-titulo { font-weight: 600; font-size: 1rem; }
.noticia { border-top: 1px solid #E8E8E1; padding: 14px 0; }
.noticia b { font-size: 1rem; display: block; }
.noticia p { margin: 6px 0 0; font-size: .9rem; line-height: 1.5; color: #363C45; }
.meta { font-size: .82rem; color: #4A515B; margin-top: 4px; }
.subtitulo { font-size: .8rem; font-weight: 700; text-transform: uppercase;
             letter-spacing: .04em; color: #4A515B; margin: 22px 0 6px; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 14px;
        border-color: #E2E2DA; }
.stButton > button { min-height: 44px; border-radius: 10px; font-weight: 600; }
@media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
</style>
"""


def inyectar_estilos() -> None:
    """Aplica el CSS global una vez por ejecución."""
    st.markdown(CSS, unsafe_allow_html=True)
