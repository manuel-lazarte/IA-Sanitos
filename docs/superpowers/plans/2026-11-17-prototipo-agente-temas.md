# Prototipo Agente de Descubrimiento de Temas (Grupo 15) — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototipo académico funcional: app Streamlit "Explorador Inteligente de Temas – Grupo 15" que carga noticias en español, las preprocesa, entrena LDA (conteos) y NMF (TF-IDF) reales con scikit-learn, muestra temas/palabras/noticias representativas, los compara y evalúa (diversidad + coherencia NPMI), con pruebas pytest que demuestren el funcionamiento.

**Architecture:** 6 módulos Python desacoplados (`src/`) orquestados por `src/pipeline.py` y consumidos por una única página Streamlit (`app.py`) con navegación por sidebar y estado en `st.session_state`. Corpus sintético generado por script determinista, claramente marcado como demostrativo.

**Tech Stack:** Python 3.12+ · Streamlit · scikit-learn (LatentDirichletAllocation, NMF) · CountVectorizer / TfidfVectorizer · pandas · numpy · matplotlib · spaCy (`es_core_news_sm`) · pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-prototipo-agente-temas-design.md` (diseño aprobado; síntesis del informe `Primer_avance_IA_Grupo_15_BLANCO_Y_NEGRO.docx`).

## Global Constraints

- Español en toda la UI y los mensajes de error. Nombre de la app: **Explorador Inteligente de Temas – Grupo 15**.
- Sin servicios de pago, sin claves API, sin LLM externos. Todo corre local (Windows/VS Code en el enunciado; esta máquina macOS para verificar).
- LDA consume **conteos** (CountVectorizer); NMF consume **TF-IDF** (TfidfVectorizer). Ambos sobre el mismo corpus preprocesado, parámetros de vocabulario idénticos (`min_df=2, max_df=0.9, max_features=3000`).
- Semilla aleatoria fija `SEMILLA_ALEATORIA = 42`, mostrada en la UI. Entrenamientos deterministas (LDA `learning_method="batch"`, NMF `init="nndsvda"`).
- **No preservar tildes está prohibido**: el preprocesamiento NO elimina tildes (informe 3.6: «público»/«publicó»). Lemmatización con spaCy preservando acentos.
- Datos sintéticos claramente identificados: medios ficticios "Medio Demo 1"/"Medio Demo 2", URLs bajo `https://demo.ejemplo/`. Nunca atribuir textos a medios reales. ~30 noticias, 6 temas.
- Sin resultados simulados: cada número mostrado en la UI sale de un modelo entrenado en la sesión.
- BERTopic NO se implementa (informe: ampliación opcional).
- Default `n_temas = 6`, `n_palabras = 10`, ajustables en la UI.
- Repo: commits locales de git (sin remote, sin push) tras cada tarea.

## Review Focus

1. **CSV con encabezados renombrados** ("Título", "URL" con mayúscula/tilde): esperado que cargue normalizando encabezados; si falta una columna requerida, error claro nombrándola, no traceback. → test en Task 3.
2. **Corpus sin vocabulario útil tras preprocesar** (todo stopwords/números): esperado `CorpusInvalidoError` con mensaje accionable, no excepción cruda de sklearn. → test en Task 5.
3. **Más temas que documentos** (o corpus mínimo): esperado `CorpusInsuficienteError` sugiriendo reducir `n_temas`. → test en Task 6.
4. **Tildes y eñes**: esperado que el vocabulario conserve acentos y que «público» y «publicó» NO se fusionen. → test en Task 4.
5. **Duplicados/republicaciones y documentos vacíos**: esperado que se eliminen con informe numérico visible. → test en Task 3.

---

### Task 0: Verificación de entorno

**Files:** none (probe only).

- [ ] **Step 1: Verificar que la instalación en segundo plano terminó OK**

Run: `.venv/bin/python -c "import sklearn, pandas, numpy, streamlit, spacy, pytest, matplotlib; print('OK')"`
Expected: `OK`. Si spaCy falla por falta de wheels para Python 3.15: recrear venv con `python3.12` (`brew list python@3.12` o el disponible) y documentar en README que Windows debe usar Python 3.11–3.13. No continuar sin este paso verde.

- [ ] **Step 2: Descargar modelo spaCy**

Run: `.venv/bin/python -m spacy download es_core_news_sm`
Expected: descarga OK y `.venv/bin/python -c "import es_core_news_sm"` sin error.

- [ ] **Step 3: git init local**

Run: `git init && git add -A && git commit -m "chore: informe y plan del proyecto"` (excluir `.venv` y `.tmp_informe` vía `.gitignore` creada en Task 1; este primer commit puede quedar pendiente hasta Task 1 si conviene).

### Task 1: Estructura del proyecto

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`, `src/__init__.py`, `tests/conftest.py`, `datos/`, `docs/superpowers/specs/2026-11-17-prototipo-agente-temas-design.md` (guardar allí el diseño aprobado)

**Interfaces:**
- Produces: `conftest.py` define `ROOT = Path(__file__).resolve().parents[1]`, inserta `ROOT` en `sys.path`, y fixtures:
  - `ruta_demo() -> Path` → `ROOT / "datos" / "noticias_demo.csv"`
  - `corpus_mini() -> list[str]` → 8 textos cortos en español: 4 del tema combustibles («El precio del diésel subió en Cochabamba. El transporte pidió abastecimiento de gasolina.»…), 4 del tema deportes («El equipo ganó el partido en La Paz. El jugador anotó dos goles en el torneo.»…)

- [ ] **Step 1: Escribir los archivos de estructura**

`requirements.txt` (pins mínimos, sin hashing):
```
streamlit>=1.40
scikit-learn>=1.6
pandas>=2.2
numpy>=2.1
matplotlib>=3.9
spacy>=3.8
pytest>=8.3
```
`pytest.ini`: `[pytest] testpaths = tests`. `.gitignore`: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.tmp_informe/`.

- [ ] **Step 2: Verificar pytest corre vacío**

Run: `.venv/bin/python -m pytest`
Expected: `no tests ran` (exit 5 es aceptable aquí).

- [ ] **Step 3: Commit** `chore: estructura del proyecto`

### Task 2: Corpus sintético de demostración

**Files:**
- Create: `datos/generar_noticias_demo.py`, `datos/noticias_demo.csv` (generado), `datos/LEEME.txt`
- Test: `tests/test_datos_demo.py`

**Interfaces:**
- Produces: CSV con columnas exactas `titulo,medio,fecha,url,contenido`; 30 filas (5 por cada uno de 6 temas: combustibles, agua/sequía, economía/precios, deportes, salud, educación); medios solo `"Medio Demo 1"` / `"Medio Demo 2"`; fechas "2026-11-XX"; urls `https://demo.ejemplo/<tema>/<nnn>`. Generador determinista (`random.Random(2026)`), invocable como `python datos/generar_noticias_demo.py`.

- [ ] **Step 1: Escribir test estructural (fallando)**

```python
def test_csv_demo_estructura(ruta_demo):
    df = pd.read_csv(ruta_demo)
    assert list(df.columns) == ["titulo", "medio", "fecha", "url", "contenido"]
    assert len(df) == 30
    assert set(df["medio"]) <= {"Medio Demo 1", "Medio Demo 2"}
    assert df["contenido"].str.split().str.len().min() >= 30
    assert df["contenido"].nunique() == 30
    assert df["url"].str.startswith("https://demo.ejemplo/").all()

def test_csv_demo_vocabulario_tematico(ruta_demo):
    texto = " ".join(pd.read_csv(ruta_demo)["contenido"]).lower()
    for palabra in ["combustible", "diésel", "gasolina", "sequía", "precio", "dólar", "partido", "vacuna", "estudiante", "escuela"]:
        assert palabra in texto
```

- [ ] **Step 2: Ejecutar test, verificar fallo** — `pytest tests/test_datos_demo.py` → FAIL (no existe el CSV).

- [ ] **Step 3: Implementar generador + LEEME**

Banco de frases por tema (≥10 frases temáticas y ≥2 títulos por tema) + frases relleno compartidas; 5 noticias por tema alternando medios y fechas de noviembre 2026; `LEEME.txt` declarando carácter sintético/demostrativo y comando de regeneración.

- [ ] **Step 4: Ejecutar tests, verificar paso** — PASS ambos.

- [ ] **Step 5: Commit** `feat: corpus sintético de 30 noticias demo`

### Task 3: Carga de datos

**Files:**
- Create: `src/carga_datos.py`
- Test: `tests/test_carga_datos.py`

**Interfaces:**
- Produces:
  - `class DatosInvalidosError(ValueError)`
  - `COLUMNAS_REQUERIDAS: tuple[str, ...] = ("titulo", "medio", "fecha", "url", "contenido")`
  - `cargar_csv(ruta: str | Path) -> tuple[pd.DataFrame, dict]` — normaliza encabezados (strip, lower, sin tildes), valida columnas (raise `DatosInvalidosError` nombrando las faltantes), descarta filas con `contenido` inválido (< 5 palabras) y duplicados exactos de `contenido`/`url` (conserva el primero). Devuelve `(df_limpio, informe)` con informe `{"filas_leidas", "invalidos", "duplicados", "validos"}`.
  - `cargar_carpeta_txt(ruta_carpeta: str | Path) -> tuple[pd.DataFrame, dict]` — un documento por `*.txt` (titulo = nombre de archivo, medio = `"Archivo TXT"`, fecha = `""`, url = `""`), misma limpieza.

- [ ] **Step 1: Escribir tests fallando** — casos: CSV demo carga 30 filas con informe `validos=30`; CSV con encabezados "Título/Medio/Fecha/URL/Contenido" carga igual (normalización); CSV sin columna `contenido` → `DatosInvalidosError` y el mensaje menciona "contenido"; CSV con 1 fila vacía + 1 duplicada → informe cuenta `invalidos=1, duplicados=1`; `cargar_carpeta_txt` con 3 .txt en `tmp_path` devuelve 3 filas.

- [ ] **Step 2: Verificar fallo** — `pytest tests/test_carga_datos.py` → FAIL (módulo no existe).

- [ ] **Step 3: Implementar** firma según Interfaces; normalización de encabezados con `unicodedata.normalize("NFKD", h).encode("ascii","ignore")`.

- [ ] **Step 4: Verificar paso.**

- [ ] **Step 5: Commit** `feat: carga y validación de documentos (CSV/TXT)`

### Task 4: Preprocesamiento en español

**Files:**
- Create: `src/preproceso.py`
- Test: `tests/test_preproceso.py`

**Interfaces:**
- Consumes: nada de tareas previas (usa spaCy).
- Produces:
  - `obtener_nlp()` — carga `es_core_news_sm` con `lru_cache`; si falta, raise con mensaje que indique `python -m spacy download es_core_news_sm`.
  - `limpiar_texto(texto: str) -> str` — minúsculas, sin URLs/emails, conserva solo letras latinas con tildes + espacios, colapsa espacios.
  - `tokenizar(texto: str) -> list[str]`
  - `procesar_texto(texto: str) -> str` — pipeline completo: limpiar → spaCy → filtrar (is_alpha, no stopword, len ≥ 3, no stopwords editoriales) → lemas unidos por espacio.
  - `procesar_corpus(textos: list[str]) -> list[str]` — batch con `nlp.pipe`.
  - `es_documento_valido(texto: str, min_palabras: int = 5) -> bool`
  - `STOPWORDS_EXTRA: frozenset[str]` — stopwords editoriales propias (p. ej. "según", "además", "dijo", "hoy", "tras").
- Reglas fijas: NO eliminar tildes; minúsculas después de limpiar, lematización preserva acentos.

- [ ] **Step 1: Escribir tests fallando** — `limpiar_texto("El PIB… https://x.com ¡Atención!")` sin URL ni signos; `procesar_texto("Las exportaciones subieron")` no contiene "las"/"subieron" y contiene "exportación" (lema verificado con el modelo instalado); **tildes**: `procesar_texto("El público publicó datos")` mantiene palabras distintas (el lema de "público" ≠ lema de "publicó"); `procesar_texto("El diésel es caro")` contiene "diésel"; `es_documento_valido("")` y `("Hola")` falsos, noticia demo verdadera; `procesar_corpus` de 3 textos devuelve 3 strings.

- [ ] **Step 2: Verificar fallo** — FAIL (módulo no existe).

- [ ] **Step 3: Implementar** — regex de limpieza `[^a-záéíóúüñ\s]+` sobre minúsculas; filtro spaCy por token (`is_alpha`, `not is_stop`, `len >= 3`, lemma ∉ STOPWORDS_EXTRA).

- [ ] **Step 4: Verificar paso.**

- [ ] **Step 5: Commit** `feat: preprocesamiento español con spaCy (lemas, tildes preservadas)`

### Task 5: Representación numérica

**Files:**
- Create: `src/representacion.py`
- Test: `tests/test_representacion.py`

**Interfaces:**
- Consumes: `corpus_mini` (conftest), textos procesados de Task 4.
- Produces:
  - `class CorpusInvalidoError(ValueError)`
  - `PARAMS_VOCABULARIO = {"min_df": 2, "max_df": 0.9, "max_features": 3000}`
  - `construir_matriz_conteos(textos: list[str]) -> tuple[CountVectorizer, csr_matrix]`
  - `construir_matriz_tfidf(textos: list[str]) -> tuple[TfidfVectorizer, csr_matrix]`
  - `terminos_vocabulario(vectorizer) -> list[str]`
  - Ambas usan `PARAMS_VOCABULARIO`; envuelven ValueError de sklearn en `CorpusInvalidoError` con mensaje accionable. Comentario en el código: conteos = entrada probabilística de LDA; TF-IDF = ponderación para NMF (informe 3.7).

- [ ] **Step 1: Escribir tests fallando** — sobre 8 textos de `corpus_mini` procesados: matrices con shape `(8, V)`, `V == len(terminos_vocabulario)`, `V >= 5`; conteos enteros ≥ 0; TF-IDF con valores > 0 y ≤ 1; vocabularios comparables: `set(count) == set(tfidf)` (mismo corpus y parámetros); corpus trivial `["palabra", "palabra", "palabra"]` (min_df=2 lo admite pero un solo término distinto) → shape (3,1) válido; corpus de solo stopwords `["de la", "de la", "de la"]` → `CorpusInvalidoError` con "vocabulario" en el mensaje.

- [ ] **Step 2: Verificar fallo.**

- [ ] **Step 3: Implementar** ambas funciones (misma firma, vectorizer distinto) + envoltura de error.

- [ ] **Step 4: Verificar paso.**

- [ ] **Step 5: Commit** `feat: representaciones CountVectorizer (LDA) y TF-IDF (NMF)`

### Task 6: Modelos LDA y NMF

**Files:**
- Create: `src/modelos.py`
- Test: `tests/test_modelos.py`

**Interfaces:**
- Consumes: matrices de Task 5.
- Produces:
  - `SEMILLA_ALEATORIA: int = 42`
  - `class CorpusInsuficienteError(ValueError)`
  - `entrenar_lda(matriz, n_temas: int = 6, semilla: int = SEMILLA_ALEATORIA) -> LatentDirichletAllocation` — `learning_method="batch"`, `max_iter=30`, `random_state=semilla`; valida `matriz.shape[0] >= n_temas` y `matriz.nnz > 0` antes de entrenar (si no, `CorpusInsuficienteError`).
  - `entrenar_nmf(matriz, n_temas: int = 6, semilla: int = SEMILLA_ALEATORIA) -> NMF` — `init="nndsvda"`, `max_iter=500`, `random_state=semilla`; mismas validaciones.
  - `palabras_por_tema(modelo, vocabulario: list[str], n_palabras: int = 10) -> list[list[str]]` — usa `modelo.components_` (válido para ambos).
  - `pesos_documento(modelo, matriz) -> np.ndarray` shape `(n_docs, n_temas)` — `lda.transform` / `nmf.transform`.
  - `normalizar_pesos(pesos: np.ndarray) -> np.ndarray` — filas que suman 1 (para comparar con LDA; los pesos crudos de NMF no son probabilidades: comentar en el código, informe 3.11).
  - `tema_predominante(pesos) -> np.ndarray` (argmax por fila).
  - `indices_representativos(pesos, n: int = 3) -> list[list[int]]` — índices de los n docs con mayor peso por tema.

- [ ] **Step 1: Escribir tests fallando** — sobre `corpus_mini` (8 docs, n_temas=3): LDA entrena y `pesos_documento` filas suman ≈1; NMF entrena, `components_ >= 0` y `pesos_documento` shape (8,3) con valores ≥ 0 (no necesariamente suman 1 antes de normalizar); `palabras_por_tema` devuelve 3 listas de 3 palabras strings existentes en el vocabulario; `tema_predominante` valores en [0,3); `indices_representativos(..., n=2)` → 3 listas de 2 índices válidos; `entrenar_lda(matriz, n_temas=10)` → `CorpusInsuficienteError` con "n_temas" en el mensaje; **reproducibilidad**: entrenar LDA dos veces con la misma semilla da `palabras_por_tema` idénticas.

- [ ] **Step 2: Verificar fallo.**

- [ ] **Step 3: Implementar.**

- [ ] **Step 4: Verificar paso.**

- [ ] **Step 5: Commit** `feat: entrenamiento LDA y NMF con semilla fija`

### Task 7: Evaluación (diversidad + coherencia NPMI)

**Files:**
- Create: `src/evaluacion.py`
- Test: `tests/test_evaluacion.py`

**Interfaces:**
- Consumes: `palabras_por_tema`, `pesos_documento`, `tema_predominante` de Task 6.
- Produces:
  - `diversidad_tematica(temas: list[list[str]]) -> float` — términos únicos / términos totales (0.0 si vacío).
  - `coherencia_npmi(temas: list[list[str]], textos_procesados: list[str], n_palabras: int = 10, ventana: int = 10) -> float` — NPMI media de pares contiguos de las top-n_palabras de cada tema; ventanas deslizantes de `ventana` tokens sobre el propio corpus; suavizado `1e-12`; devuelve el promedio sobre todos los temas.
  - `conteo_predominante(pesos) -> dict[int, int]` — nº de docs por tema predominante, claves 0-based ordenadas.
  - `resumen_modelo(nombre: str, modelo, matriz, textos_procesados, n_palabras: int = 10) -> dict` — `{"nombre", "n_temas", "temas": [{"tema": id0, "palabras": [...], "coherencia": float}], "diversidad": float, "coherencia": float, "distribucion": {tema: n}}`.
- Comentario en el código citando informe 3.13 (misma medida y mismo corpus para comparar modelos).

- [ ] **Step 1: Escribir tests fallando** — diversidad: 2 temas con palabras disjuntas → 1.0; 2 temas idénticos → 0.5; lista vacía → 0.0. Coherencia: corpus sintético con dos grupos de palabras fuertemente co-ocurrentes (4 docs «diésel combustible precio transporte», 4 docs «gol equipo partido torneo») → `coherencia_npmi(temas_del_corpus) > coherencia_npmi(temas_aleatorias_fijas)` donde `temas_aleatorias_fijas` son literales cruzadas (diésel con gol, etc.); coherencia acotada (-1, 1). `conteo_predominante` sobre pesos de LDA de `corpus_mini` suma 8. `resumen_modelo` devuelve claves exactas y `len(temas) == matriz`-temas.

- [ ] **Step 2: Verificar fallo.**

- [ ] **Step 3: Implementar** — NPMI: `pmi = log(p12/(p1*p2+eps))`, `npmi = pmi / (-log(p12+eps))`; conteo binario por ventana.

- [ ] **Step 4: Verificar paso.**

- [ ] **Step 5: Commit** `feat: evaluación con diversidad temática y coherencia NPMI`

### Task 8: Pipeline end-to-end + prueba de integración

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_integracion.py`

**Interfaces:**
- Consumes: Tasks 2–7 (todas las interfaces anteriores).
- Produces:
  - `ejecutar_pipeline(df: pd.DataFrame, n_temas: int = 6, n_palabras: int = 10, semilla: int = SEMILLA_ALEATORIA) -> dict` con claves `{"df", "textos_procesados", "conteos", "tfidf", "lda", "nmf", "resumenes"}`; `conteos`/`tfidf` son tuplas `(vectorizer, matriz)`; `resumenes = {"LDA": resumen, "NMF": resumen}`. Valida `len(df) >= n_temas` con `CorpusInsuficienteError`.

- [ ] **Step 1: Escribir test de integración fallando** — cargar CSV demo → `ejecutar_pipeline(df, n_temas=6)`: 6 temas por modelo, cada lista con 10 palabras; **los temas reflejan el corpus**: la unión de palabras de ambos modelos contiene al menos 8 de `{"combustible","diésel","gasolina","agua","sequía","precio","dólar","partido","vacuna","estudiante"}`; diversidad > 0.5 en ambos; distribución por tema predominante suma 30; **determinismo**: ejecutar el pipeline dos veces produce `palabras_por_tema` idénticas en ambos modelos (prueba de que no hay resultados simulados: todo sale del entrenamiento).

- [ ] **Step 2: Verificar fallo.**

- [ ] **Step 3: Implementar `ejecutar_pipeline`** (composición directa de los módulos; sin lógica nueva).

- [ ] **Step 4: Verificar paso** — este test valida que el corpus sintético del Task 2 produce temas separables; si falla por calidad del corpus, mejorar bancos de frases del generador (no relajar el test).

- [ ] **Step 5: Commit** `feat: pipeline end-to-end + prueba de integración sobre corpus demo`

### Task 9: Aplicación Streamlit

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `ejecutar_pipeline`, `cargar_csv`, `cargar_carpeta_txt`, `resumen_modelo` (vía pipeline), informes de carga.
- Produces:
  - Helpers puros testeables: `tabla_temas(resumen: dict) -> pd.DataFrame` (columnas `Tema, Palabras, Noticias predominantes, Coherencia`, Tema mostrado 1-based) y `etiqueta_tema(id_tema: int, palabras: list[str], n: int = 5) -> str` → `"Tema 3 · combustible, diésel, precio, transporte, gasolina"`.
  - `main()` ejecuta la app; guard `if __name__ == "__main__": main()` (Streamlit ejecuta el script como `__main__`).
- UI (todo en español, sidebar con radio): **Inicio** (descripción del proyecto, áre/subárea, aviso de datos demo); **1. Cargar noticias** (uploader CSV/TXT-múltiples o botón "usar corpus demo", tabla de documentos, informe de carga con eliminados/duplicados); **2. Preprocesamiento** (botón procesar; tabla original vs procesada de 3 ejemplos; n.º de documentos válidos); **3. Modelo LDA** (slider n_temas 2–10, botón entrenar, temas con palabras y pesos, gráfico de barras matplotlib de distribución predominante, noticias representativas por tema con título/medio/fecha); **4. Modelo NMF** (igual + nota "los pesos de NMF no son probabilidades"); **5. Comparación y evaluación** (tabla lado a lado de ambos resúmenes, diversidad y coherencia por modelo, gráfico comparativo). `st.session_state["pipeline"]` guarda el dict del pipeline; badge "Semilla aleatoria: 42" en las páginas de modelos; caption "Datos demostrativos sintéticos" cuando el corpus cargado sea el demo.

- [ ] **Step 1: Escribir tests fallando** — `tabla_temas` sobre un resumen literal: columnas exactas, fila por tema, Tema 1-based; `etiqueta_tema(2, ["a","b","c","d","e"], 3) == "Tema 3 · a, b, c"` (ojo: muestra id+1); `import app` no levanta la UI (guard `__main__`).

- [ ] **Step 2: Verificar fallo.**

- [ ] **Step 3: Implementar `app.py` completo** (páginas según Interfaces; gráficos matplotlib con `plt.subplots` y `st.pyplot(fig)`, cerrando cada `fig`).

- [ ] **Step 4: Verificar paso** (pytest) **+ smoke manual**: `streamlit run app.py --server.headless true` en background, `curl -s http://localhost:8501 | head -5` devuelve HTML, luego matar el proceso.

- [ ] **Step 5: Commit** `feat: app Streamlit Explorador Inteligente de Temas – Grupo 15`

### Task 10: README y documentación de defensa

**Files:**
- Create: `README.md`, `docs/explicacion_academica.md`

**Interfaces:** ninguna (documentación).

- [ ] **Step 1: README.md** — qué es el proyecto (con link al informe), requisitos (Python 3.11–3.13, aviso de compatibilidad), instalación en Windows + VS Code (`python -m venv .venv`, activar, `pip install -r requirements.txt`, `python -m spacy download es_core_news_sm`), ejecución (`streamlit run app.py`), cómo correr las pruebas (`pytest` con qué comportamiento esperar por prueba: 8 grupos del spec), cómo regenerar el corpus demo, aviso de datos sintéticos.

- [ ] **Step 2: docs/explicacion_academica.md** — las 12 preguntas del spec §9 respondidas en lenguaje sencillo, cada una anclada al informe (sección correspondiente) y al código (módulo que lo implementa). Cita las referencias del informe (Blei 2003/2012, Röder 2015, Dieng 2020, Manning 2008).

- [ ] **Step 3: Verificación final completa** — `pytest` (todos los tests pasan, contar cuántos), `streamlit run` smoke de nuevo, y revisar que `datos/LEEME.txt` + README declaran el carácter sintético.

- [ ] **Step 4: Commit** `docs: README y guía académica para la defensa`
