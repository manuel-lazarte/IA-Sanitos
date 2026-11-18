# Explorador Inteligente de Temas – Grupo 15

**Prototipo académico** de agente de análisis documental: descubre **temas
recurrentes** en noticias en español mediante **aprendizaje no supervisado**
(modelado temático), comparando **LDA** (conteos de palabras) y **NMF**
(TF-IDF) sobre el mismo corpus. Universidad Mayor de San Simón (UMSS) ·
Ingeniería de Sistemas · Inteligencia Artificial · Gestión II-2026.

> Este prototipo implementa el primer avance del grupo (ver
> `Primer_avance_IA_Grupo_15_BLANCO_Y_NEGRO.docx`): preprocesamiento en
> español, representación numérica (CountVectorizer / TF-IDF), entrenamiento
> real de LDA y NMF con scikit-learn, evaluación por **diversidad temática** y
> **coherencia NPMI**, e interfaz Streamlit. BERTopic queda como ampliación
> futura, según el informe.

## ⚠️ Datos de demostración

El corpus incluido (`datos/noticias_demo.csv`, 30 noticias) es **100%
sintético**: textos ficticios generados con semilla fija, medios
inexistentes («Medio Demo 1» / «Medio Demo 2») y URLs de ejemplo
(`https://demo.ejemplo/`). **Ningún texto proviene de un medio real.**
Los 6 temas intencionales del corpus son: combustibles, agua/sequía,
economía/precios, deportes, salud y educación. Ver `datos/LEEME.txt`.

## Requisitos

- **Python 3.11 a 3.13** (recomendado en Windows). El proyecto también fue
  verificado en Python 3.15 (macOS).
- Sin claves API ni servicios externos: todo corre localmente.
- Conexión solo necesaria una vez para instalar dependencias y el modelo
  de spaCy.

## Instalación en Windows (VS Code)

```bash
# 1. Crear entorno virtual (en la terminal integrada de VS Code)
python -m venv .venv

# 2. Activar (PowerShell)
.venv\Scripts\Activate.ps1
#    o en CMD:  .venv\Scripts\activate.bat

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el modelo de español de spaCy (obligatorio, ~13 MB)
python -m spacy download es_core_news_sm
```

Si PowerShell bloquea la activación: `Set-ExecutionPolicy -ExecutionPolicy
RemoteSigned -Scope CurrentUser` (una sola vez).

## Ejecución

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. Flujo de uso:

1. **Cargar noticias** — botón «Cargar» con el corpus demo, o sube tu CSV
   (columnas `titulo, medio, fecha, url, contenido`) o archivos TXT.
2. **Preprocesamiento** — botón «Procesar textos» (spaCy: minúsculas, sin
   URLs/signos, sin palabras vacías, lemas; **tildes preservadas**).
3. **Modelo LDA** / **4. Modelo NMF** — elige el número de temas (2–10) y
   entrena. La semilla es fija (42): los resultados son reproducibles.
4. **Comparación y evaluación** — diversidad y coherencia NPMI de ambos
   modelos lado a lado.

## Pruebas

```bash
pytest
```

Comportamiento esperado (44 pruebas en 8 archivos):

| Archivo | Qué comprueba |
|---|---|
| `test_datos_demo.py` | estructura y vocabulario del corpus demo (30 noticias, medios sintéticos) |
| `test_carga_datos.py` | carga CSV/TXT, normalización de encabezados, error si falta columna, conteo de inválidos/duplicados |
| `test_preproceso.py` | limpieza, lematización, stopwords, **tildes preservadas** («público» ≠ «publicó»), detección de documentos vacíos |
| `test_representacion.py` | matrices de conteos y TF-IDF, vocabularios comparables, error si no hay vocabulario útil |
| `test_modelos.py` | entrenamiento LDA/NMF, distribuciones, reproducibilidad con semilla, error si hay más temas que documentos |
| `test_evaluacion.py` | diversidad temática y coherencia NPMI (temas coherentes > pares cruzados) |
| `test_integracion.py` | pipeline completo sobre el corpus demo: 6 temas por modelo, temas que reflejan el corpus, **determinismo** (dos ejecuciones = mismos temas) |
| `test_app.py` | helpers de presentación de la interfaz |

Salida esperada: `44 passed` en unos segundos.

## Regenerar el corpus demo

```bash
python datos/generar_noticias_demo.py
```

## Estructura del proyecto

```
IA/
├── app.py                     # Interfaz Streamlit (ETAPA 5, presentación)
├── src/
│   ├── carga_datos.py         # ETAPA 1: carga CSV/TXT, duplicados, vacíos
│   ├── preproceso.py          # ETAPA 2: limpieza, lemas, stopwords (spaCy)
│   ├── representacion.py      # ETAPA 3: CountVectorizer (LDA) y TF-IDF (NMF)
│   ├── modelos.py             # ETAPA 4: entrenamiento LDA y NMF (scikit-learn)
│   ├── evaluacion.py          # ETAPA 6: diversidad + coherencia NPMI
│   └── pipeline.py            # Orquestador: une las seis etapas
├── datos/
│   ├── noticias_demo.csv      # Corpus sintético (30 noticias, 6 temas)
│   ├── generar_noticias_demo.py
│   └── LEEME.txt              # Aviso de datos sintéticos
├── tests/                     # Pruebas pytest (8 archivos, 44 pruebas)
├── docs/
│   ├── explicacion_academica.md   # Guía para la exposición/defensa
│   └── superpowers/           # Diseño y plan de implementación
├── requirements.txt
└── pytest.ini
```

## Documentación académica

Para la exposición y defensa del proyecto, ver
[`docs/explicacion_academica.md`](docs/explicacion_academica.md): qué es el
agente, cómo funciona LDA y NMF, qué cumple TF-IDF, cómo se comparan los
modelos y qué demuestra este prototipo — cada respuesta anclada al informe
y al código.

## Límites del prototipo

- El corpus demo es sintético y pequeño (30 noticias); los temas describen
  **esa muestra**, no la prensa boliviana.
- El corpus real (200–400 noticias de hasta dos medios, con uso autorizado)
  se construirá en etapas posteriores del proyecto.
- La interpretación de los temas (nombres legibles) es posterior a los
  modelos y requiere revisión humana, como establece el informe.
