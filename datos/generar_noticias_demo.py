"""Generador del corpus sintético de demostración - Grupo 15.

Genera 30 noticias 100% FICTICIAS en español (5 por cada uno de 6 temas),
con medios ficticios "Medio Demo 1" / "Medio Demo 2" y URLs de ejemplo bajo
https://demo.ejemplo/. Ningún texto proviene de un medio real ni debe
atribuirse a uno.

Uso:
    python datos/generar_noticias_demo.py

El resultado es determinista (semilla 2026) y se escribe en
datos/noticias_demo.csv junto a este script.
"""
import random
from pathlib import Path

import pandas as pd

SEMILLA = 2026
N_POR_TEMA = 5
CARPETA = Path(__file__).resolve().parent
RUTA_CSV = CARPETA / "noticias_demo.csv"
RUTA_LEEME = CARPETA / "LEEME.txt"

# ---------------------------------------------------------------- temas
TEMAS = {
    "combustibles": {
        "titulos": [
            "Escasez de diésel complica el transporte de carga",
            "Filas en estaciones por falta de gasolina",
            "Anuncian nuevos despachos de combustible para el agro",
            "El subsidio al combustible, en el centro del debate",
            "Control reforzado contra el contrabando de combustibles",
        ],
        "frases": [
            "El precio del diésel subió esta semana en las estaciones de servicio del país.",
            "El transporte pesado denunció retrasos en el abastecimiento de combustible en Cochabamba.",
            "Los conductores esperan largas filas por la escasez de gasolina en La Paz.",
            "El abastecimiento de combustible se normalizó en las principales ciudades del oriente.",
            "La empresa estatal anunció nuevos despachos de diésel para el sector agropecuario.",
            "El subsidio al combustible concentra el debate sobre el presupuesto nacional.",
            "Delegaciones del sector transporte dialogaron sobre el suministro de gasolina especializada.",
            "El contrabando de combustible preocupa a las autoridades de control en la frontera.",
            "Transportistas de Santa Cruz exigieron garantías de abastecimiento de diésel para la cosecha.",
            "El precio internacional afecta el costo de la gasolina en el mercado interno.",
        ],
    },
    "agua_sequia": {
        "titulos": [
            "Sequía pone en riesgo el abastecimiento de agua",
            "Agricultores cuentan pérdidas por la falta de lluvias",
            "Declaran emergencia hídrica en zonas altas",
            "Comunidades organizan turnos de riego",
            "Embalses en su nivel más bajo del año",
        ],
        "frases": [
            "La sequía reduce los niveles de las represas que abastecen de agua a Cochabamba.",
            "Los agricultores piden apoyo frente a la pérdida de cultivos por la falta de lluvias.",
            "El municipio declaró emergencia hídrica por la sequía prolongada en las zonas altas.",
            "Comunidades del altiplano organizan turnos de riego ante la escasez de agua.",
            "El embalse principal registra su nivel más bajo de los últimos años.",
            "Especialistas recomiendan cosechar agua de lluvia para mitigar la sequía.",
            "La escasez de agua afecta a miles de familias en los valles del país.",
            "El gobierno presentó un plan de emergencia por la sequía en el altiplano.",
            "Los comunarios construyen atajados para almacenar agua en época de lluvias.",
            "La falta de lluvias retrasa la siembra en las zonas productoras.",
        ],
    },
    "economia_precios": {
        "titulos": [
            "El dólar se mantiene estable, pero con presión",
            "Suben los precios de la canasta familiar",
            "Comerciantes ajustan precios por el tipo de cambio",
            "Analistas advierten sobre las reservas internacionales",
            "Alzas en la carne y el pollo golpean los mercados",
        ],
        "frases": [
            "El precio del dólar se mantiene estable en el mercado paralelo esta semana.",
            "La inflación acumulada preocupa a las familias por el alza de los precios de los alimentos.",
            "Los comerciantes ajustan los precios ante la variación del tipo de cambio.",
            "El banco central intervino para estabilizar el precio del dólar.",
            "El costo de la canasta familiar subió en los mercados de La Paz y El Alto.",
            "Los exportadores piden un tipo de cambio competitivo para sostener las ventas.",
            "El precio del pan y del arroz aumentó en los mercados populares.",
            "Analistas advierten presión sobre el dólar por la caída de las reservas.",
            "Los mercados de abasto registran alzas en el precio de la carne y el pollo.",
            "El gobierno revisa medidas para frenar el alza de precios de productos básicos.",
        ],
    },
    "deportes": {
        "titulos": [
            "El equipo local ganó y sigue líder del torneo",
            "Gol en el último minuto clasifica al club",
            "La selección ya piensa en el partido clave",
            "El clásico se jugará con estadio lleno",
            "Arranca el campeonato nacional de fútbol",
        ],
        "frases": [
            "El equipo local ganó el partido por dos goles a uno en el estadio.",
            "El delantero anotó un gol decisivo para la clasificación al torneo internacional.",
            "La selección nacional entrenó de cara al partido eliminatorio de este mes.",
            "El clásico del fútbol boliviano se jugará con estadio lleno el fin de semana.",
            "El club contrató a un nuevo entrenador para afrontar el torneo de apertura.",
            "Los hinchas celebraron el triunfo del equipo en las calles del centro.",
            "El arquero atajó un penal en el último minuto del partido.",
            "El torneo de fútbol infantil reunió a cientos de niños en el sur de la ciudad.",
            "La jugadora fue la figura del partido con dos goles anotados.",
            "El campeonato nacional de clubes arranca con nuevos cupos para el fútbol regional.",
        ],
    },
    "salud": {
        "titulos": [
            "Comienza la campaña de vacunación contra el dengue",
            "Bajan los casos de enfermedades respiratorias",
            "Hospitales amplían su emergencia nocturna",
            "Brigadas de salud recorren barrios periféricos",
            "Advierten aumento de diabetes en jóvenes",
        ],
        "frases": [
            "Los centros de salud iniciaron la campaña de vacunación contra el dengue.",
            "El ministerio de salud reportó una baja en los casos de enfermedades respiratorias.",
            "Los hospitales amplían sus servicios de emergencia para atender la demanda nocturna.",
            "Las brigadas de salud recorren barrios periféricos para aplicar vacunas a los niños.",
            "Los médicos advierten sobre el aumento de casos de diabetes en adultos jóvenes.",
            "El seguro gratuito cubre consultas y medicamentos en los establecimientos de salud.",
            "Autoridades inspeccionan la calidad de los alimentos para prevenir brotes de diarrea.",
            "La red de salud habilitó turnos de vacunación los fines de semana.",
            "Profesionales de salud capacitan a promotores comunitarios en primeros auxilios.",
            "El hospital del niño recibió equipos nuevos para su sala de cuidados intensivos.",
        ],
    },
    "educacion": {
        "titulos": [
            "Estudiantes retomaron clases en colegios públicos",
            "Entregan computadoras a escuelas rurales",
            "Maestros piden mejores condiciones para el año escolar",
            "La universidad abre nuevas carreras tecnológicas",
            "Feria de ciencias reúne a estudiantes destacados",
        ],
        "frases": [
            "Los estudiantes de colegios públicos retomaron las clases después de las festividades.",
            "El ministerio de educación entregó computadoras a escuelas rurales del país.",
            "Los maestros piden mejores condiciones para el inicio del año escolar.",
            "La universidad amplió su oferta académica con nuevas carreras tecnológicas.",
            "Los padres de familia gestionan el mantenimiento de la escuela del barrio.",
            "El gobierno promete conectividad gratuita para los estudiantes de zonas alejadas.",
            "La feria de ciencias reunió a estudiantes destacados de todo el departamento.",
            "Las escuelas incorporan herramientas digitales para reforzar el aprendizaje de las matemáticas.",
            "Los docentes participan en cursos de actualización sobre educación inclusiva.",
            "El programa de desayuno escolar se amplía a más unidades educativas.",
        ],
    },
}

# Frases de relleno editorial compartidas (aportan vocabulario común del corpus).
CONECTORES = [
    "La información fue confirmada por fuentes oficiales consultadas en la capital.",
    "El reporte indica que las autoridades siguen de cerca la situación.",
    "Versiones de los involucrados serán incorporadas cuando existan declaraciones formales.",
    "Se espera un pronunciamiento oficial en las próximas horas.",
]

MEDIOS = ["Medio Demo 1", "Medio Demo 2"]


def generar() -> pd.DataFrame:
    """Construye el DataFrame de 30 noticias sintéticas (determinista)."""
    rng = random.Random(SEMILLA)
    filas = []
    indice = 0
    for tema, datos in TEMAS.items():
        usados: set[str] = set()
        for n, titulo in enumerate(datos["titulos"]):
            # 4 frases temáticas distintas + 1 conector compartido.
            while True:
                frases = rng.sample(datos["frases"], 4)
                contenido = " ".join(frases + [rng.choice(CONECTORES)])
                if contenido not in usados:
                    usados.add(contenido)
                    break
            filas.append(
                {
                    "titulo": titulo,
                    "medio": MEDIOS[indice % len(MEDIOS)],
                    "fecha": f"2026-11-{N_POR_TEMA * 6 - indice:02d}",
                    "url": f"https://demo.ejemplo/{tema}/{indice + 1:03d}",
                    "contenido": contenido,
                }
            )
            indice += 1
    return pd.DataFrame(filas, columns=["titulo", "medio", "fecha", "url", "contenido"])


def escribir_leeme() -> None:
    texto = """CORPUS SINTETICO DE DEMOSTRACION - GRUPO 15 (IA, UMSS)
==========================================================

Las 30 noticias de noticias_demo.csv son COMPLETAMENTE FICTICIAS:
fueron generadas por este script (semilla 2026) para probar el
prototipo. Los medios "Medio Demo 1" y "Medio Demo 2" no existen,
las URLs apuntan al dominio de ejemplo demo.ejemplo y ningun texto
proviene de un medio real.

Temas intencionales (5 noticias por tema): combustibles, agua/sequia,
economia/precios, deportes, salud, educacion.

Regenerar el corpus:
    python datos/generar_noticias_demo.py

El corpus real del proyecto (200-400 noticias de medios bolivianos,
con uso autorizado) se construira en etapas posteriores segun el
informe del grupo.
"""
    RUTA_LEEME.write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    df = generar()
    df.to_csv(RUTA_CSV, index=False, encoding="utf-8")
    escribir_leeme()
    print(f"CSV sintético escrito en {RUTA_CSV} ({len(df)} noticias)")
