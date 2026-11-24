# Informe de fuentes y calidad — Corpus real de noticias bolivianas

**Proyecto:** IA-Sanitos · Grupo 15 · UMSS · Aprendizaje no supervisado / PLN
**Rama de trabajo:** `T2-cristian-manuel` · **Fecha de recopilación:** 2026-11-23

## 1. Objetivo

Construir la colección inicial de noticias bolivianas reales (informe 4.3,
Tabla 1: entre 200 y 400 artículos, máximo dos medios) que alimenta las
rutas LDA / NMF del proyecto, con procedencia documentada por registro.

## 2. Medios investigados y decisión

| Medio | robots.txt | Acceso verificado | Decisión |
|---|---|---|---|
| **ABI** (agencia estatal, La Paz) | `Disallow:` vacío (permite todo) | sitemaps WordPress OK | **SELECCIONADO** |
| **El Deber** (diario, Santa Cruz) | `Allow: /` en artículos; excluye `/buscar/`, `/service/` (API), PDFs | sitemaps OK | **SELECCIONADO** |
| La Razón | 403 para clientes automatizados | bloqueado | descartado (no eludir bloqueos) |
| Los Tiempos | sin respuesta | inaccesible | descartado |
| Unitel | robots OK | contenido audiovisual, texto corto | reserva (límite de 2 medios) |

**Justificación de la selección:** los dos seleccionados permiten la
consulta de artículos por mecanismos públicos expresamente publicados
(sitemaps), con texto completo en español, producción diaria alta y
naturalezas editoriales distintas (agencia estatal nacional vs. diario
privado regional), lo que aporta diversidad dentro del límite de dos medios.

## 3. Condiciones de uso

- Los textos pertenecen a sus respectivos medios; el corpus se conserva
  únicamente para uso académico interno del Grupo 15 (análisis, docencia,
  defensa). No se redistribuye ni republica.
- Cada registro conserva su URL original para verificar la fuente.
- No se usaron APIs excluidas por robots.txt (p. ej. `/service/` de El
  Deber), ni se eludieron paywalls o bloqueos.
- User-Agent identificable del recolector y pausa de cortesía de 0,8 s
  entre descargas.

## 4. Metodología

1. Auditoría de acceso (robots.txt + sitemaps de los 6 medios candidatos).
2. Prueba piloto (3 artículos por medio) con verificación de titulares,
   fechas y calidad del cuerpo extraído.
3. Recopilación por sitemaps públicos con muestreo uniforme en el
   intervalo (diversidad temporal) y semilla fija (reproducible).
4. Filtrado: secciones periodísticas (sin opinión, publicidad, sindicaciones
   BBC ni navegación), mínimo 30 palabras, deduplicación por URL y huella
   SHA-1 del contenido.

## 5. Procedencia y campos

Cada registro lleva: id estable, titular, medio, fecha de publicación,
URL original, fecha de recopilación, sección original, autor (o ausencia
documentada), procedimiento de obtención, nº palabras y huella de
contenido — ver `trazabilidad_manifiesto.csv`.

## 6. Resultados de la recopilación

Contadores reales de la ejecución (ver `informe_recopilacion.json` y las
bitácoras):

| Etapa | ABI | El Deber |
|---|---|---|
| Candidatos en sitemaps (dentro del intervalo) | 924 | 127–138 |
| Descargas de artículos | 320 | 127 |
| Fallos de descarga | 0 | 1 |
| **Válidos finales** | **262** (tras thinning uniforme 320→262) | **126** |
| **TOTAL DEL CORPUS** | **388 registros** | |

- Distribución temporal ABI: 17 días (1–17 nov), ~15 noticias/día.
- El Deber: 21–23 nov (los únicos días que exponen sus sitemaps públicos).
- Duplicados exactos en el corpus final: 0 (verificados por huella SHA-1).
- Longitud: media 378 palabras, mediana 322, mínimo 47.
- Auditoría manual: 4 registros al azar verificados EN VIVO contra sus
  URLs originales — los 4 titulares COINCIDEN (2026-11-23).

## 7. Limpieza aplicada

- Extracción limitada al contenedor del artículo (sin menús, CSS ni JS).
- Eliminación de avisos de interfaz y cierres comerciales.
- Colapso de espacios; conservación de tildes, ñ y puntuación original.
- Sin reescritura ni traducción de contenidos; sin completar textos.

## 8. Validaciones ejecutadas

- Carga real con `src/carga_datos.py` (el cargador del prototipo).
- `validar_dataset.py`: columnas, obligatorios, unicidad, fechas ISO,
  URLs, duplicados por huella, residuos HTML, prueba de vectorización
  TF-IDF.
- Pruebas automáticas en `tests/test_dataset_real.py` (rango del informe,
  máximo dos medios, campos completos, manifiesto coincidente).

## 9. Limitaciones

- Intervalos desiguales por medio: ABI 1–17 de noviembre (el muestreo no
  alcanzó la cola 18–23 de sus candidatos) y El Deber 21–23 de noviembre
  (límite de sus sitemaps públicos; su archivo/API está excluido por
  robots.txt y no se utilizó).
- La fecha de El Deber corresponde a publicación/última modificación según
  sus propios metadatos.
- Corpus de dos medios: los temas describen la cobertura de ABI y El Deber
  en el periodo, no toda la prensa boliviana.
- La Razón bloqueó el acceso automatizado (403) y Los Tiempos no respondió:
  evaluados y descartados en la auditoría inicial.

## 10. Recomendaciones de integración

- Entrenar SIEMPRE sobre `contenido` (la sección editorial no es etiqueta).
- Separar demo (`datos/noticias_demo.csv`) del corpus real: nunca mezclar.
- Frente A (modelos): puede consumir `noticias_bolivia.csv` directamente.
- Frente C (sesgos): usar `trazabilidad_manifiesto.csv` (medio, fecha,
  sección) para el estudio por periodo y medio.
