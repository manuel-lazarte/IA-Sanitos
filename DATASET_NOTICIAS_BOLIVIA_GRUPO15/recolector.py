"""Recolector del corpus real de noticias bolivianas - Grupo 15 (Frente B).

Obtiene artículos auténticos de DOS medios (máximo del informe 4.3) usando
EXCLUSIVAMENTE mecanismos que los propios sitios publican para descubrimiento
(sitemaps) y respetando sus robots.txt:

- ABI (agencia estatal, La Paz): robots.txt permite todo ('Disallow:' vacío).
  Candidatos: post-sitemapN.xml (WordPress), intervalo 1-23 nov 2026.
- El Deber (diario privado, Santa Cruz): robots.txt 'Allow: /' para artículos;
  excluye /buscar/, /portadas/, /service/ (su API) y PDFs. Candidatos:
  sitemap-organico.xml + sitemap-news.xml, intervalo 21-23 nov 2026
  (LIMITACIÓN documentada: sus sitemaps solo exponen los últimos días).

Uso:
    python recolector.py --piloto   # 3 artículos por medio, verificación
    python recolector.py            # recopilación completa (~300 válidos)

Salidas (en esta carpeta):
    noticias_bolivia.csv        5 campos compatibles con src/carga_datos.py
    trazabilidad_manifiesto.csv todos los metadatos de auditoría por registro
    informe_recopilacion.json   contadores reales de candidatos/descartes
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd

CARPETA = Path(__file__).resolve().parent
USER_AGENT = "IA-Sanitos-Grupo15/1.0 (proyecto academico UMSS; coleccion de corpus para PLN)"
PAUSA_SEGUNDOS = 0.8  # cortesía con los servidores
OBJETIVO_POR_MEDIO = 160
MIN_PALABRAS = 30

INTERVALO_ABI = ("2026-11-01", "2026-11-24")
INTERVALO_DEBER = ("2026-11-21", "2026-11-24")

# Secciones de El Deber con contenido periodístico (no opinión, no publicidad,
# no sindicaciones BBC, no navegación). Permite lista cerrada y auditable.
SECCIONES_DEBER = {
    "bolivia", "pais", "santa-cruz", "economia", "dinero", "deportes", "futbol",
    "multideportivo", "tenis", "tecnologia", "salud-y-bienestar",
    "educacion-y-sociedad", "cultura-y-entretenimiento", "al-dia", "noticias",
    "mundo", "clima", "sostenibilidad-y-negocios", "universidades",
}


def _descargar(url: str) -> str:
    """Descarga con User-Agent identificable, una pausa de cortesía y 1 reintento."""
    peticion = urlrequest.Request(url, headers={"User-Agent": USER_AGENT})
    for intento in range(2):
        try:
            with urlrequest.urlopen(peticion, timeout=25) as respuesta:
                return respuesta.read().decode("utf-8", errors="replace")
        except urlerror.HTTPError as e:
            if e.code in (429, 500, 502, 503) and intento == 0:
                time.sleep(3 * (intento + 1))
                continue
            return ""
        except (urlerror.URLError, TimeoutError, OSError):
            if intento == 0:
                time.sleep(2)
                continue
            return ""


def _limpiar_html(fragmento: str) -> str:
    sin_etiquetas = re.sub(r"<[^>]+>", " ", fragmento)
    sin_entidades = (
        sin_etiquetas.replace("&nbsp;", " ").replace("&amp;", "&")
        .replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")
        .replace("&#8211;", "–").replace("&#8220;", "“").replace("&#8221;", "”")
    )
    return re.sub(r"\s+", " ", sin_entidades).strip()


def _jsonld(html: str) -> dict:
    """Primer bloque JSON-LD de tipo NewsArticle/Article (mejor esfuerzo)."""
    for bloque in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S):
        try:
            datos = json.loads(bloque)
        except json.JSONDecodeError:
            continue
        elementos = datos if isinstance(datos, list) else [datos]
        for e in elementos:
            if isinstance(e, dict) and str(e.get("@type", "")).lower() in ("newsarticle", "article"):
                return e
    return {}


# ----------------------------------------------------------------- sitemaps
def candidatos_abi(inicio: str = INTERVALO_ABI[0]) -> list[dict]:
    """URLs de post-sitemapN con lastmod dentro del intervalo."""
    candidatos = []
    for numero in (11, 10):  # sitemap11 (más nuevo) y sitemap10
        xml = _descargar(f"https://abi.bo/post-sitemap{numero}.xml")
        pares = re.findall(r"<loc>([^<]+)</loc>\s*(?:<lastmod>([^<]+)</lastmod>)?", xml)
        for url, lastmod in pares:
            fecha = (lastmod or "")[:10]
            if re.match(r"^\d{4}-\d{2}-\d{2}$", fecha) and inicio <= fecha < INTERVALO_ABI[1]:
                candidatos.append({"url": url, "lastmod": fecha})
    return candidatos


def candidatos_deber() -> list[dict]:
    """URLs de artículo de los sitemaps públicos, dentro del intervalo."""
    candidatos = []
    vistos = set()
    for sitemap in ("sitemap-organico.xml", "sitemap-news.xml"):
        xml = _descargar(f"https://eldeber.com.bo/{sitemap}")
        for url, fecha in re.findall(r"<loc>([^<]+)</loc>\s*(?:<lastmod>([^<]+)</lastmod>)?", xml):
            if url in vistos:
                continue
            vistos.add(url)
            seccion = url.split("/")[3] if url.count("/") >= 3 else ""
            if seccion not in SECCIONES_DEBER:
                continue
            fecha_dia = (fecha or "")[:10]
            if fecha_dia and not (INTERVALO_DEBER[0] <= fecha_dia < INTERVALO_DEBER[1]):
                continue
            candidatos.append({"url": url, "lastmod": fecha_dia, "seccion": seccion})
    return candidatos


# ------------------------------------------------------- extracción por medio
def extraer_deber(html: str, url: str) -> dict | None:
    datos = _jsonld(html)
    titulo = datos.get("headline") or ""
    m = re.search(r'<meta property="og:title" content="([^"]*)"', html)
    titulo = titulo or (m.group(1) if m else "")
    fecha = datos.get("datePublished") or ""
    autor = ""
    if isinstance(datos.get("author"), dict):
        autor = datos["author"].get("name", "")
    seccion = url.split("/")[3]

    # Cuerpo: solo los párrafos del contenedor del artículo (evita CSS/JS y
    # los avisos de interfaz que cuelgan del mismo contenedor).
    frases_ui = ("¿Quiere recibir notificaciones", "Iniciar sesión", "Cerrar sesión")
    cuerpo = ""
    corte = html.find("articulo__cuerpo")
    if corte != -1:
        zona = html[corte : corte + 120_000]
        parrafos = [_limpiar_html(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", zona, re.S)]
        parrafos = [
            p for p in parrafos
            if len(p) > 40 and "{" not in p and "window." not in p and "function" not in p
            and not any(p.startswith(frase) for frase in frases_ui)
        ]
        cuerpo = "\n".join(parrafos)
    if len(cuerpo.split()) < MIN_PALABRAS or not titulo or not fecha:
        return None
    return {
        "titulo": _limpiar_html(titulo), "fecha": fecha[:10], "autor": autor,
        "seccion": seccion, "contenido": cuerpo,
    }


def extraer_abi(html: str, url: str) -> dict | None:
    datos = _jsonld(html)
    titulo = datos.get("headline") or ""
    m = re.search(r'<meta property="og:title" content="([^"]*)"', html)
    titulo = titulo or (m.group(1) if m else "")
    m = re.search(r'<meta property="article:published_time" content="([^"]*)"', html)
    fecha = (datos.get("datePublished") or (m.group(1) if m else ""))[:10]
    autor = ""
    if isinstance(datos.get("author"), dict):
        autor = datos["author"].get("name", "")

    # Cuerpo: dentro de la etiqueta <article> del tema WordPress de ABI.
    cuerpo = ""
    inicio = html.find("<article")
    fin = html.find("</article>")
    if inicio != -1 and fin != -1 and fin > inicio:
        zona = html[inicio:fin]
        parrafos = [_limpiar_html(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", zona, re.S)]
        parrafos = [
            p for p in parrafos
            if len(p) > 40
            and not p.startswith("Agencia Boliviana de Información –")
            and "Síganos" not in p and "Lea también" not in p
        ]
        cuerpo = "\n".join(parrafos)
    if len(cuerpo.split()) < MIN_PALABRAS or not titulo or not fecha:
        return None
    return {
        "titulo": _limpiar_html(titulo), "fecha": fecha, "autor": autor,
        "seccion": "nacional", "contenido": cuerpo,
    }


# ----------------------------------------------------------------- principal
def recopilar(
    piloto: bool = False,
    abi_inicio: str = INTERVALO_ABI[0],
    anexar: bool = False,
    objetivo_total: int = 320,
    cuotas_por_medio: dict | None = None,
) -> None:
    random.seed(2026)  # muestreo reproducible
    huella = lambda texto: hashlib.sha1(" ".join(texto.split()).encode()).hexdigest()[:16]
    contadores: dict[str, int] = {}
    registros, trazas = [], []

    # Modo anexar: continúa un corpus existente sin repetir URLs (p. ej. para
    # ampliar el intervalo de ABI hacia octubre y llegar al tope del informe).
    if anexar:
        ruta_csv = CARPETA / "noticias_bolivia.csv"
        ruta_traza = CARPETA / "trazabilidad_manifiesto.csv"
        if ruta_csv.exists() and ruta_traza.exists():
            previos = pd.read_csv(ruta_csv, encoding="utf-8")
            trazas_previas = pd.read_csv(ruta_traza, encoding="utf-8")
            registros = previos.to_dict("records")
            trazas = trazas_previas.to_dict("records")
            contadores["previos_cargados"] = len(registros)
    urls_vistas = {r["url"] for r in registros}
    huellas = {
        hashlib.sha1(" ".join(r["contenido"].split()).encode()).hexdigest()[:16]
        for r in registros
    }
    objetivo_restante = objetivo_total - len(registros)

    fuentes = [
        ("ABI", "https://abi.bo", lambda: candidatos_abi(abi_inicio), extraer_abi, (abi_inicio, INTERVALO_ABI[1])),
        ("El Deber", "https://eldeber.com.bo", candidatos_deber, extraer_deber, INTERVALO_DEBER),
    ]

    for medio, base, obtener_candidatos, extraer, (inicio, fin) in fuentes:
        if piloto:
            cuota_medio = 3
        elif cuotas_por_medio and medio in cuotas_por_medio:
            cuota_medio = cuotas_por_medio[medio]
        else:
            cuota_medio = max(0, objetivo_total - len(registros))
        if cuota_medio == 0:
            contadores[f"{medio}:omitido_cuota_completa"] = 0
            continue
        candidatos = obtener_candidatos()
        contadores[f"{medio}:candidatos"] = len(candidatos)
        # Muestreo uniforme a lo largo del intervalo (diversidad temporal).
        candidatos.sort(key=lambda c: c.get("lastmod") or "")
        paso = max(1, len(candidatos) mmmmmmmmnbn.     // (cuota_medio + 5))
        muestra = candidatos[::paso][: cuota_medio + 10]
        validos_medio = 0
        for c in muestra:
            if validos_medio >= cuota_medio or len(registros) >= objetivo_total:
                break
            if c["url"] in urls_vistas:
                contadores[f"{medio}:ya_en_corpus"] = contadores.get(f"{medio}:ya_en_corpus", 0) + 1
                continue
            html = _descargar(c["url"])
            contadores[f"{medio}:descargas"] = contadores.get(f"{medio}:descargas", 0) + 1
            if not html:
                contadores[f"{medio}:fallos_descarga"] = contadores.get(f"{medio}:fallos_descarga", 0) + 1
                continue
            articulo = extraer(html, c["url"])
            if articulo is None:
                contadores[f"{medio}:descartados_extraccion"] = contadores.get(f"{medio}:descartados_extraccion", 0) + 1
                continue
            h = huella(articulo["contenido"])
            if h in huellas or c["url"] in urls_vistas:
                contadores[f"{medio}:duplicados"] = contadores.get(f"{medio}:duplicados", 0) + 1
                continue
            huellas.add(h)
            urls_vistas.add(c["url"])
            validos_medio += 1
            registros.append(
                {
                    "id_noticia": f"{medio.split()[0].lower()}-{h[:10]}",
                    "titulo": articulo["titulo"],
                    "medio": medio,
                    "fecha": articulo["fecha"],
                    "url": c["url"],
                    "contenido": articulo["contenido"],
                }
            )
            trazas.append(
                {
                    "id_noticia": registros[-1]["id_noticia"],
                    "url": c["url"],
                    "medio": medio,
                    "fecha_publicacion": articulo["fecha"],
                    "fecha_recopilacion": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "seccion_original": articulo["seccion"],
                    "autor": articulo["autor"] or "(sin autor publicado)",
                    "procedimiento": "sitemap publico + descarga de pagina del articulo",
                    "palabras": len(articulo["contenido"].split()),
                    "huella_contenido": h,
                }
            )
            time.sleep(PAUSA_SEGUNDOS)
        contadores[f"{medio}:validos"] = validos_medio

    if piloto:
        for r in registros:
            print(f"[{r['medio']}] {r['fecha']} | {r['titulo'][:70]}")
            print(f"   {r['url']}")
            print(f"   palabras={len(r['contenido'].split())} | inicio: {r['contenido'][:110]}...")
        print("\nCONTADORES:", json.dumps(contadores, ensure_ascii=False, indent=2))
        return

    # Entregables (extrasaction=ignore: los registros llevan id_noticia
    # para el manifiesto; el CSV del cargador solo usa los 5 campos).
    with open(CARPETA / "noticias_bolivia.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=["titulo", "medio", "fecha", "url", "contenido"],
            extrasaction="ignore",
        )
        escritor.writeheader()
        escritor.writerows(registros)
    with open(CARPETA / "trazabilidad_manifiesto.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(trazas[0].keys()))
        escritor.writeheader()
        escritor.writerows(trazas)
    with open(CARPETA / "informe_recopilacion.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fecha_recopilacion": datetime.now(timezone.utc).isoformat(),
                "medios": ["ABI", "El Deber"],
                "intervalos": {"ABI": INTERVALO_ABI, "El Deber": INTERVALO_DEBER},
                "objetivo": 300,
                "modo_anexar": anexar,
                "cuotas_por_medio": cuotas_por_medio,
                "validos": len(registros),
                "contadores": contadores,
                "limitaciones": [
                    "La Razón bloquea clientes automatizados (403) y Los Tiempos no "
                    "respondió: se descartaron en la auditoría de acceso.",
                    "Los sitemaps públicos de El Deber solo exponen los últimos días; "
                    "su archivo/API está excluido en robots.txt y NO se utilizó.",
                    "La fecha de El Deber proviene del lastmod/JSON-LD del artículo; "
                    "corresponde a publicación o última modificación.",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(contadores, ensure_ascii=False, indent=2))
    print(f"TOTAL VÁLIDOS: {len(registros)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--piloto", action="store_true")
    parser.add_argument("--anexar", action="store_true",
                        help="continúa el corpus existente sin repetir URLs")
    parser.add_argument("--abi-inicio", default=INTERVALO_ABI[0],
                        help="fecha de inicio del intervalo de ABI (AAAA-MM-DD)")
    parser.add_argument("--objetivo", type=int, default=320,
                        help="total de artículos válidos buscado")
    parser.add_argument("--cuota-abi", type=int, default=None,
                        help="cuota fija de artículos para ABI")
    parser.add_argument("--cuota-deber", type=int, default=None,
                        help="cuota fija de artículos para El Deber")
    args = parser.parse_args()
    cuotas = {}
    if args.cuota_abi is not None:
        cuotas["ABI"] = args.cuota_abi
    if args.cuota_deber is not None:
        cuotas["El Deber"] = args.cuota_deber
    recopilar(
        piloto=args.piloto,
        abi_inicio=args.abi_inicio,
        anexar=args.anexar,
        objetivo_total=args.objetivo,
        cuotas_por_medio=cuotas or None,
    )
