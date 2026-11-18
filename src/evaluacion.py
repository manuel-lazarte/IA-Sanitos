"""ETAPA 6 del agente: evaluación de los temas descubiertos - Grupo 15.

Métricas del informe 3.13:

- DIVERSIDAD TEMÁTICA: proporción de términos únicos entre las listas de
  palabras principales de todos los temas (1.0 = ningún término repetido).

- COHERENCIA NPMI: asociación estadística entre las palabras principales
  de cada tema, medida por co-aparición en ventanas del PROPIO corpus.
  NPMI normaliza el PMI al rango (-1, 1). Se aplica EXACTAMENTE la misma
  medida, corpus y número de términos a LDA y a NMF, como exige el informe
  para comparar modelos (Röder et al., 2015; Dieng et al., 2020).

La revisión humana (nombres de temas y lectura de noticias representativas)
complementa estas métricas y se realiza en la interfaz, no aquí.
"""
from collections import Counter
from itertools import combinations

import numpy as np

from src.modelos import palabras_por_tema, tema_predominante

EPS = 1e-12


def diversidad_tematica(temas: list[list[str]]) -> float:
    """Términos únicos / términos totales entre las listas principales."""
    if not temas:
        return 0.0
    total = sum(len(t) for t in temas)
    if total == 0:
        return 0.0
    unicos = {p for t in temas for p in t}
    return len(unicos) / total


def _ventanas_por_documento(textos_procesados: list[str], ventana: int):
    """Ventanas DESLIZANTES (paso 1) de `ventana` tokens por documento.

    Igual que las implementaciones estándar de coherencia (gensim,
    Palmetto): cada palabra participa en varias ventanas solapadas, lo que
    da muchos más casos de co-aparición que trozos no solapados.
    """
    ventanas = []
    for texto in textos_procesados:
        tokens = texto.split()
        if len(tokens) <= ventana:
            if len(tokens) >= 2:
                ventanas.append(tokens)
            continue
        for inicio in range(len(tokens) - ventana + 1):
            ventanas.append(tokens[inicio : inicio + ventana])
    return ventanas


def _estadisticas_coocurrencia(textos_procesados: list[str], ventana: int):
    """Conteos de co-aparición para NPMI: ventanas deslizantes (paso 1) y
    pares NO ordenados de palabras distintas (segmentación S_one-one de
    Röder et al., 2015).

    Devuelve (total_ventanas, n_pares, n_individuales): cuántas ventanas
    contienen cada par y cada palabra.
    """
    n_pares: Counter = Counter()
    n_individual: Counter = Counter()
    total_ventanas = 0
    for trozo in _ventanas_por_documento(textos_procesados, ventana):
        total_ventanas += 1
        unicos = sorted(set(trozo))
        n_individual.update(unicos)
        n_pares.update(combinations(unicos, 2))
    return total_ventanas, n_pares, n_individual


def _npmi_par(w1: str, w2: str, total: int, n_pares, n_individual) -> float:
    """NPMI de un par: PMI dividido por -log p(w1, w2), acotado a (-1, 1)."""
    p12 = (n_pares.get(tuple(sorted((w1, w2))), 0) + EPS) / total
    p1 = (n_individual.get(w1, 0) + EPS) / total
    p2 = (n_individual.get(w2, 0) + EPS) / total
    return float(np.log(p12 / (p1 * p2)) / (-np.log(p12)))


def coherencia_npmi(
    temas: list[list[str]],
    textos_procesados: list[str],
    n_palabras: int = 10,
    ventana: int = 10,
) -> float:
    """NPMI media de las palabras principales de todos los temas.

    Misma medida para LDA y NMF: mismos temas, mismo corpus de referencia
    y mismo número de términos destacados.
    """
    total, n_pares, n_individual = _estadisticas_coocurrencia(
        textos_procesados, ventana
    )
    if total == 0:
        return 0.0
    valores = []
    for tema in temas:
        palabras = tema[:n_palabras]
        for w1, w2 in combinations(palabras, 2):
            valores.append(_npmi_par(w1, w2, total, n_pares, n_individual))
    if not valores:
        return 0.0
    return float(np.mean(valores))


def conteo_predominante(pesos: np.ndarray) -> dict[int, int]:
    """Número de documentos cuyo tema predominante es cada tema (0-based)."""
    predominantes = tema_predominante(pesos)
    return {int(t): int(n) for t, n in sorted(Counter(predominantes).items())}


def resumen_modelo(
    nombre: str,
    modelo,
    matriz,
    vocabulario: list[str],
    textos_procesados: list[str],
    n_palabras: int = 10,
) -> dict:
    """Evaluación completa de un modelo entrenado, lista para la interfaz.

    Claves: nombre, n_temas, temas [{tema, palabras, coherencia}],
    diversidad, coherencia (media), distribucion (docs por tema predominante).
    """
    temas = palabras_por_tema(modelo, vocabulario, n_palabras)
    pesos = modelo.transform(matriz)
    return {
        "nombre": nombre,
        "n_temas": len(temas),
        "temas": [
            {
                "tema": k,
                "palabras": tema,
                "coherencia": _coherencia_tema(tema, textos_procesados, n_palabras),
            }
            for k, tema in enumerate(temas)
        ],
        "diversidad": diversidad_tematica(temas),
        "coherencia": coherencia_npmi(temas, textos_procesados, n_palabras),
        "distribucion": conteo_predominante(pesos),
    }


def _coherencia_tema(
    tema: list[str], textos_procesados: list[str], n_palabras: int, ventana: int = 10
) -> float:
    """NPMI media de los pares de palabras principales de UN tema."""
    total, n_pares, n_individual = _estadisticas_coocurrencia(
        textos_procesados, ventana
    )
    if total == 0:
        return 0.0
    pares = list(combinations(tema[:n_palabras], 2))
    if not pares:
        return 0.0
    valores = [_npmi_par(w1, w2, total, n_pares, n_individual) for w1, w2 in pares]
    return float(np.mean(valores))
