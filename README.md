# Descubrimiento de temas en noticias bolivianas

[![Pruebas](https://github.com/manuel-lazarte/IA-Sanitos/actions/workflows/tests.yml/badge.svg)](https://github.com/manuel-lazarte/IA-Sanitos/actions/workflows/tests.yml)

**UMSS · Ingeniería de Sistemas · Inteligencia Artificial · Grupo 15 · II-2026**

IA-Sanitos es un sistema de aprendizaje no supervisado que identifica temas recurrentes en noticias en español sin utilizar etiquetas predefinidas. Permite explorar palabras clave, noticias representativas y métricas mediante una interfaz web local.

## Métodos utilizados

| Método | Funcionamiento |
|---|---|
| LDA | Identifica mezclas probabilísticas de temas a partir de conteos de palabras. |
| NMF | Factoriza una matriz TF-IDF para obtener componentes temáticos. |
| BERTopic | Genera embeddings con Multilingual E5, reduce dimensiones con UMAP, agrupa por densidad con HDBSCAN y describe los temas mediante c-TF-IDF. |

LDA y NMF requieren seleccionar el número de temas. BERTopic descubre grupos sin fijar ese número previamente; las noticias sin grupo se identifican con `-1`.

## Tecnologías

- Python
- Pandas y NumPy
- spaCy
- Scikit-learn
- Sentence Transformers
- BERTopic
- UMAP
- HDBSCAN
- Streamlit
- Matplotlib
- Pytest

## Datos

El proyecto dispone de dos corpus independientes:

| Corpus | Cantidad | Tipo |
|---|---:|---|
| ABI | 262 noticias | Real |
| El Deber | 126 noticias | Real |
| Corpus real total | 388 noticias | Real |
| Corpus de demostración | 30 noticias | Sintético |

**Ubicación de los datasets:**

- Corpus real: `DATASET_NOTICIAS_BOLIVIA_GRUPO15/noticias_bolivia.csv`
- Corpus de demostración: `datos/noticias_demo.csv`

El archivo CSV debe contener las siguientes columnas:

| Columna | Descripción |
|---|---|
| `titulo` | Título de la noticia. |
| `medio` | Medio de comunicación. |
| `fecha` | Fecha de publicación. |
| `url` | Enlace de la noticia original. |
| `contenido` | Cuerpo textual de la noticia. |

También se admiten archivos TXT, uno por noticia.

Los corpus real y sintético no deben mezclarse. Los textos reales están sujetos a las condiciones de uso de sus respectivos medios.

Para más información, consultar la [documentación del dataset](DATASET_NOTICIAS_BOLIVIA_GRUPO15/LEEME.md).

## Instalación

### Requisitos

- Python 3.11 recomendado.
- Git.
- Conexión a internet para instalar las dependencias y descargar los modelos.

No se necesita GPU ni clave API.

### Windows (PowerShell)

```powershell
git clone https://github.com/manuel-lazarte/IA-Sanitos.git
cd IA-Sanitos

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt -r requirements-bertopic.txt

python -m spacy download es_core_news_sm

python -m pip check
```

### Linux / macOS

```bash
git clone https://github.com/manuel-lazarte/IA-Sanitos.git
cd IA-Sanitos

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt -r requirements-bertopic.txt

python -m spacy download es_core_news_sm

python -m pip check
```

Si las dependencias declaradas están correctamente instaladas y son compatibles, el último comando mostrará:

```text
No broken requirements found.
```

La ruta completa de BERTopic cuenta con evidencia de verificación en Windows con Python 3.11.9. La compatibilidad de una instalación limpia puede variar según el sistema operativo y la arquitectura del equipo.

## Ejecución

Con el entorno virtual activado, ejecutar desde la raíz del repositorio:

```bash
python -m streamlit run app.py
```

Acceder a la aplicación desde el navegador:

http://localhost:8501

### Funcionamiento

1. **Cargar noticias:** seleccionar el corpus de demostración o cargar un archivo CSV o archivos TXT.
2. **Analizar:** seleccionar el número de temas para LDA y NMF (6 por defecto), mantener activada la opción **Incluir BERTopic** y pulsar **Analizar noticias**.
3. **Explorar temas:** seleccionar el método, consultar las palabras principales, noticias representativas y métricas, y aplicar filtros por medio o fecha.

La primera ejecución de BERTopic requiere descargar el codificador Multilingual E5.

Para ejecutar los tres métodos directamente desde Python, se debe indicar `modelos=("LDA", "NMF", "BERTopic")` al llamar a `ejecutar_pipeline`.

## Evaluación

El sistema utiliza los siguientes indicadores:

- **Coherencia NPMI:** mide la asociación estadística entre las palabras principales de los temas de LDA y NMF.
- **Diversidad temática:** mide la proporción de términos únicos entre los temas descubiertos.
- **Cobertura temática:** permite identificar la proporción de noticias agrupadas y sin grupo en BERTopic.

La evaluación NPMI comparable de BERTopic y el registro de revisión humana se encuentran pendientes.

## Estructura del repositorio

```text
IA-Sanitos/
├── app.py                   # Interfaz Streamlit
├── src/
│   ├── carga_datos.py       # Carga y validación
│   ├── preproceso.py        # Limpieza y lematización
│   ├── representacion.py    # Conteos y TF-IDF
│   ├── modelos.py           # LDA y NMF
│   ├── bertopic_modelo.py   # E5, UMAP, HDBSCAN y c-TF-IDF
│   ├── evaluacion.py        # Métricas de evaluación
│   ├── pipeline.py          # Coordinación de modelos
│   └── demo_bertopic.py      # Ejecución por consola
├── ui/                      # Componentes de interfaz
├── datos/                   # Corpus sintético
├── DATASET_NOTICIAS_BOLIVIA_GRUPO15/
│                            # Corpus real y trazabilidad
├── docs/                    # Documentación técnica
├── tests/                   # Pruebas automatizadas
├── requirements.txt
└── requirements-bertopic.txt
```

## Pruebas

Ejecutar las pruebas automatizadas:

```bash
python -m pytest -q
```

Validar el dataset real:

```bash
python DATASET_NOTICIAS_BOLIVIA_GRUPO15/validar_dataset.py
```

Comprobar el funcionamiento de BERTopic:

```bash
python -m src.demo_bertopic
```

La ejecución de demostración exporta sus resultados a `output/bertopic/resumen.json`.

### Pruebas de integración de BERTopic

Después de ejecutar la demostración y descargar el codificador E5, se pueden habilitar las pruebas de integración semántica.

**Windows (PowerShell):**

```powershell
$env:RUN_BERTOPIC_INTEGRATION = "1"
python -m pytest -q
Remove-Item Env:RUN_BERTOPIC_INTEGRATION
```

**Linux / macOS:**

```bash
RUN_BERTOPIC_INTEGRATION=1 python -m pytest -q
```

Para más información, consultar la [documentación técnica de BERTopic y HDBSCAN](docs/DECISION_TECNICA_BERTOPIC_HDBSCAN.md) y la [evidencia de verificación](docs/VERIFICACION_BERTOPIC_HDBSCAN.md).

## Integrantes

- Ignacio Alex Escobar Vasquez
- Manuel Jotan Lazarte Ascuy
- Cristian Samuel Veliz Huanaco
- Jhoel Zeballos Zelada
- Ivan Guido Estrada Aguilar
- Leonardo Anjhelo Quiñones
- Jack Kevin Yavi Almendras

**Docente:** Lic. Erika Patricia Rodríguez Bilbao.

**Universidad Mayor de San Simón — Facultad de Ciencias y Tecnología.**

Gestión II-2026 · Cochabamba, Bolivia.
