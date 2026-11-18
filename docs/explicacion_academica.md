# Guía académica para defender el prototipo — Grupo 15

Respuestas sencillas a las preguntas de la defensa, cada una anclada a la
sección del informe y al módulo del código que la implementa.

## 1. ¿Qué es el agente inteligente desarrollado?

Es un **agente de análisis documental**: recibe una solicitud del usuario
(cargar noticias y descubrir temas), ejecuta un procesamiento por etapas
(carga → preprocesamiento → representación → modelos → evaluación) y
devuelve resultados interpretables (temas con sus palabras y noticias
representativas). No es un chatbot generativo: no escribe textos nuevos,
solo analiza los documentos que recibe.
**Informe:** 3.1 (agentes de Russell y Norvig) y 4.2. **Código:** `src/pipeline.py` (orquestador) y `app.py` (interfaz).

## 2. ¿Qué problema resuelve?

En una colección de noticias, los asuntos concretos que se repiten (por
ejemplo, el abastecimiento de combustibles) atraviesan secciones
editoriales y usan vocabularios distintos. Leer noticia por noticia no
escala y la búsqueda por palabras exactas solo encuentra lo que coincide
literalmente. El agente agrupa por **patrones de co-aparición de
palabras**, no por sección ni por palabra exacta.
**Informe:** 1 (Introducción) y 4.1. **Código:** el corpus demo muestra un tema de combustibles que reúne noticias de "transporte", "frontera" y "agro".

## 3. ¿Por qué utiliza aprendizaje no supervisado?

Porque **no existen etiquetas temáticas** asignadas por nadie: el sistema
no aprende de ejemplos con respuesta conocida (como haría un clasificador
supervisado de «economía» o «deportes»), sino que **explora la estructura**
de los propios documentos. Eso permite descubrir asuntos más específicos
que las secciones de un medio.
**Informe:** 3.3. **Código:** los modelos se entrenan solo con las matrices de palabras (`src/modelos.py`); ninguna columna del CSV es una etiqueta de entrenamiento.

## 4. ¿Qué es el modelado temático?

Es una técnica no supervisada que busca **estructuras recurrentes** en una
colección y las describe con términos característicos y documentos
asociados. Los temas son **latentes**: no se observan directamente, se
infieren de señales como los conteos de palabras. Una lista como
«combustible, diésel, abastecimiento, transporte» sugiere un tema que un
humano puede nombrar después.
**Informe:** 3.9. **Código:** `src/modelos.py` (descubrimiento) y la página 3/4 de la app (presentación de palabras por tema).

## 5. ¿Cómo funciona LDA?

LDA (Latent Dirichlet Allocation, Blei et al. 2003) es un **modelo
probabilístico generativo**: supone que cada documento es una **mezcla de
temas** y cada tema una **distribución sobre el vocabulario**; para cada
posición del texto se elige un tema según la mezcla del documento y se
"genera" una palabra según ese tema. Al entrenar, se observan las palabras
y se **infieren** las distribuciones. LDA trabaja con **conteos de
palabras** (bolsa de palabras, sin orden) y entrega, para cada noticia, las
**proporciones** de cada tema (valores que suman 1: son probabilidades bajo
el modelo). Por eso una noticia puede tener peso en varios temas a la vez.
**Informe:** 3.10. **Código:** `entrenar_lda` y `pesos_documento` en `src/modelos.py`; entrada generada por `construir_matriz_conteos` en `src/representacion.py`.

## 6. ¿Cómo funciona NMF?

NMF (Non-negative Matrix Factorization) aproxima la matriz TF-IDF
(documentos × términos, todos valores ≥ 0) como el **producto de dos
matrices más pequeñas, también no negativas**: una dice **cuánto contribuye
cada componente (tema) a cada documento** y la otra **qué peso tiene cada
término en cada componente**. Al inspeccionar los términos de mayor peso de
un componente, lo interpretamos como un tema. **Diferencia clave con LDA:**
sus pesos son **coeficientes de una factorización algebraica, no
probabilidades** (no vienen de un modelo generativo). La app muestra esos
pesos crudos; el módulo ofrece `normalizar_pesos()` para convertirlos en
proporciones por documento (filas que suman 1) si se quiere leerlos como
"reparto de peso", dejando claro que esa lectura es solo de presentación.
**Informe:** 3.11. **Código:** `entrenar_nmf`, `normalizar_pesos` y el comentario en `src/modelos.py`; entrada generada por `construir_matriz_tfidf`.

## 7. ¿Qué función cumple TF-IDF?

TF-IDF pondera cada palabra por **cuánto aparece en el documento** (TF) y
**qué tan rara es en toda la colección** (IDF). Así una palabra presente en
casi todas las noticias («dijo», «según») pierde peso, y una palabra
frecuente en pocas noticias («diésel») lo gana. Es la representación de
entrada de NMF; LDA, en cambio, usa conteos crudos porque su modelo
probabilístico ya maneja la frecuencia de otra forma.
**Informe:** 3.7 (Manning et al. 2008). **Código:** `construir_matriz_tfidf` y el comentario de la ruta B en `src/representacion.py`.

## 8. ¿Cómo aprende el sistema sin categorías previas?

Por **estadística de co-aparición**: las palabras que tienden a aparecer
juntas en los mismos documentos terminan en el mismo tema. En el corpus
demo, «combustible», «diésel» y «gasolina» co-aparecen en las noticias
ficticias de combustibles, así que el modelo las agrupa **sin que nadie le
diga** que eso es "un tema de combustibles". El equipo solo define el
procedimiento y la configuración (número de temas, semilla); **los temas se
estiman de los textos**.
**Informe:** 3.2 y 3.9. **Código:** `tests/test_integracion.py` prueba que los temas descubiertos reflejan los 6 temas del corpus sin pasarle ninguna etiqueta.

## 9. ¿Cómo se descubren los temas?

En seis etapas (informe 4.2): (1) se cargan y depuran noticias
(`src/carga_datos.py`: vacíos, duplicados); (2) se preprocesan en español
(`src/preproceso.py`: minúsculas, sin URLs, sin palabras vacías, lemas,
**tildes preservadas** para no fusionar «público» con «publicó»); (3) se
representan como matriz de **conteos** (ruta LDA) y matriz **TF-IDF** (ruta
NMF) con el mismo vocabulario (`src/representacion.py`); (4) se entrenan
ambos modelos con **semilla fija 42** (`src/modelos.py`); (5) cada tema se
presenta con sus 10 palabras principales, un gráfico de pesos y sus 3
noticias más representativas (`app.py`); (6) el nombre del tema se propone
**leyendo** esas salidas: el modelo entrega palabras, el humano la
interpretación.

## 10. ¿Cómo se comparan los modelos?

Con **medidas homogéneas** (informe 3.13: misma medida, mismo corpus, mismo
número de términos): **diversidad temática** (proporción de términos únicos
entre las listas principales de todos los temas) y **coherencia NPMI**
(asociación por co-aparición en ventanas del propio corpus de las palabras
principales de cada tema), calculadas de forma idéntica para LDA y NMF en
`src/evaluacion.py`. Además se compara la **distribución de documentos por
tema predominante** y, sobre todo, la **interpretabilidad**: los temas
mezclados de LDA frente a los grupos más nítidos de NMF en el corpus demo.
La pertinencia final se confirma leyendo noticias representativas.

## 11. ¿Qué demuestra este primer prototipo?

Que el **ciclo completo funciona de punta a punta con resultados reales**:
carga de CSV/TXT, preprocesamiento en español con lematización, dos
representaciones numéricas comparables, entrenamiento real de LDA y NMF,
descubrimiento de temas interpretables (con el corpus sintético de 30
noticias, **NMF recupera con nitidez los 6 temas**, uno por grupo de 5
noticias; **LDA los descubre de forma parcial y los mezcla** —p. ej.
combustibles con salud—, lo que ilustra la diferencia de ambos modelos
descrita en la pregunta 10), evaluación con métricas definidas y una
interfaz Streamlit navegable. Las 59 pruebas de `tests/` —incluida una de
**determinismo** (dos ejecuciones producen exactamente los mismos temas)—
verifican que no hay resultados simulados.

## 12. ¿Qué faltará para el proyecto final?

- **Corpus real**: 200–400 noticias de hasta dos medios bolivianos con uso
  autorizado (el actual es sintético y pequeño; sus temas describen solo
  esa muestra). Informe 4.3.
- **Barido de hiperparámetros**: ensayar varios números de temas y elegir
  con coherencia/diversidad e interpretación (informe 4.4).
- **Coherencia Cv** sobre corpus de referencia externo y protocolo de
  revisión humana con dos revisores y registro de desacuerdos (informe
  3.13).
- **BERTopic** como ampliación experimental con embeddings en español,
  registrando la proporción de documentos sin grupo (informe 3.12).
- Análisis por **medio y por fecha** (los metadatos se conservan para eso)
  y control del vocabulario dominante de un medio (informe 4.5).

## Referencias (del informe)

- Blei, Ng y Jordan (2003), *Latent Dirichlet allocation*, JMLR 3.
- Blei (2012), *Probabilistic topic models*, CACM 55(4).
- Kuang, Choo y Park (2015), *NMF for interactive topic modeling*.
- Manning, Raghavan y Schütze (2008), *Introduction to Information Retrieval*.
- Röder, Both y Hinneburg (2015), *Exploring the space of topic coherence measures*.
- Dieng, Ruiz y Blei (2020), *Topic modeling in embedding spaces*, TACL 8.
- Grootendorst (2022), *BERTopic*, arXiv:2203.05794.
- O'Callaghan et al. (2015), *An analysis of the coherence of descriptors in topic modeling*.
