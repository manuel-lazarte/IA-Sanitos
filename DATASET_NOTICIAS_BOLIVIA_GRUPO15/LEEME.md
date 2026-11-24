# DATASET NOTICIAS BOLIVIA – GRUPO 15 (corpus real)

Colección de noticias bolivianas REALES y verificables para las rutas
LDA / NMF (y BERTopic) del proyecto. **Este corpus es distinto e
independiente de las 30 noticias sintéticas** de `datos/noticias_demo.csv`
(prototipo inicial): nunca mezclar los dos conjuntos.

## Qué contiene

| Archivo | Descripción |
|---|---|
| `noticias_bolivia.csv` | Corpus en 5 campos EXACTOS del cargador del proyecto: `titulo, medio, fecha, url, contenido` |
| `trazabilidad_manifiesto.csv` | Auditoría por registro: id estable, URL, medio, fechas de publicación y recopilación, sección, autor, procedimiento, nº palabras, huella de contenido |
| `informe_recopilacion.json` | Contadores reales: candidatos, descargas, descartes, duplicados, válidos; medios, intervalos y limitaciones |
| `recolector.py` | Script de recopilación (determinista, con pausa de cortesía y UA identificable) |
| `validar_dataset.py` | Validación técnica completa (ver abajo) |
| `bitacora_recopilacion.txt` | Salida cruda de la recopilación |

## Cómo se obtuvo

Mecanismos públicos de descubrimiento de cada medio (sitemaps) + descarga
del artículo individual, respetando robots.txt de cada sitio:

- **ABI** (agencia estatal, La Paz): robots.txt permite el acceso completo
  (`Disallow:` vacío). Sitemaps `post-sitemap10/11.xml`.
- **El Deber** (diario privado, Santa Cruz): robots.txt permite artículos
  (`Allow: /`; excluye `/buscar/`, `/service/`, PDFs). Sitemaps
  `sitemap-organico.xml` + `sitemap-news.xml`.

Cada artículo se verificó extrayendo titular, fecha y cuerpo desde sus
propios metadatos (JSON-LD / etiquetas `article:published_time`).

## Intervalo y cobertura

Ver `informe_recopilacion.json` (números reales, no estimados). Limitaciones
documentadas allí: La Razón bloquea clientes automatizados (403), Los
Tiempos no respondió, y los sitemaps de El Deber solo exponen los últimos
días (su archivo/API está excluida por robots.txt y NO se utilizó).

## Restricciones de uso

Los textos pertenecen a sus medios. El dataset se conserva SOLO para uso
académico interno del Grupo 15 (UMSS, materia IA): análisis, docencia y
defensa del proyecto. No redistribuir ni republicar. Cada registro conserva
su URL original para verificar la fuente.

## Cómo importarlo en Python

```python
import pandas as pd

df = pd.read_csv("DATASET_NOTICIAS_BOLIVIA_GRUPO15/noticias_bolivia.csv")
textos = df["contenido"].tolist()      # <- campo para PLN
metadatos = df[["titulo", "medio", "fecha", "url"]]  # trazabilidad
```

Con el cargador del proyecto (mismas columnas):

```python
from src.carga_datos import cargar_csv
df, informe = cargar_csv("DATASET_NOTICIAS_BOLIVIA_GRUPO15/noticias_bolivia.csv")
print(informe)  # filas_leidas, invalidos, duplicados, validos
```

## Qué campos usar para PLN y cuáles solo para trazabilidad

- **PLN:** SOLO `contenido` (y opcionalmente `titulo` concatenado). La
  sección del medio es metadato editorial: NO es etiqueta de entrenamiento.
- **Trazabilidad/auditoría:** `titulo`, `medio`, `fecha`, `url` y todo el
  manifiesto (autor, sección, huella, fechas de recopilación).

## Validación

```bash
python DATASET_NOTICIAS_BOLIVIA_GRUPO15/validar_dataset.py
```

Revisa columnas, obligatorios, unicidad, fechas ISO, URLs, duplicados por
huella, residuos HTML y ejecuta una prueba real de vectorización TF-IDF.
Sale con código 0 solo si no hay errores.
