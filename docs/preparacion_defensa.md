# Preparación para la Defensa — Explorador Inteligente de Temas (Grupo 15)

Documento de estudio para la defensa oral ante la docente. Todas las
respuestas están ancladas al **código real** (con archivo y línea), al
**informe del primer avance** (sección citada) y a **resultados reales de
ejecución** (nada inventado). Ejecutar `pytest` y `streamlit run app.py`
antes de la defensa para tener todo fresco.

> **Cómo practicar:** estudia los bloques, luego pide la simulación oral
> (pregunta por pregunta con evaluación 1–10). Al final hay la guía para
> ese simulacro.

---

## BLOQUE A — Preguntas generales del proyecto

**1. ¿De qué trata su proyecto?**
- **Corta:** Es un agente de análisis documental que descubre automáticamente los temas recurrentes en una colección de noticias en español, sin usar categorías predefinidas. Compara dos modelos: LDA y NMF, y muestra cada tema con sus palabras y noticias más representativas.
- **Técnica:** Aplicamos *topic modeling* (modelado temático), un enfoque de aprendizaje no supervisado del área de PLN. La unidad de análisis es la noticia; cada tema se infiere de la co-aparición de palabras en los documentos, no de etiquetas humanas.
- **Ejemplo:** Con 30 noticias de prueba, el sistema descubrió solo un tema cuyas palabras son «combustible, abastecimiento, gasolina, transporte, diésel».
- **Repregunta:** «¿Y eso para qué sirve en la práctica?» → Para explorar hemerotecas: localizar grupos de noticias relacionadas sin leerlas todas, por ejemplo detectar que hay una ola de noticias sobre abastecimiento de combustibles.

**2. ¿Qué problema intenta solucionar?**
- **Corta:** Las categorías de sección de un periódico («economía», «deportes») no capturan asuntos específicos que cruzan secciones, y buscar por palabra exacta se pierde noticias que usan otro vocabulario. Nuestro agente agrupa por patrones de palabras co-ocurrentes.
- **Técnica:** El informe (secciones 1 y 4.1) lo plantea así: una consulta por «sequía» omite textos que hablan de «escasez de agua». El modelo no depende de coincidencias literales sino de co-aparición estadística.
- **Ejemplo:** Una noticia sobre «el precio del diésel» aparece en nuestro corpus bajo economía/transporte/agro, y el modelo la agrupa con las demás de combustibles sin que nadie la etiquete.
- **Repregunta:** «¿Y no hace eso un buscador?» → No: el buscador necesita que escribas la palabra; el agente te dice qué temas EXISTEN en la colección, que es lo que no sabías de antemano.

**3. ¿Cuál es el objetivo principal?**
- **Corta:** Diseñar y evaluar un procedimiento exploratorio no supervisado para identificar y describir temas recurrentes en noticias bolivianas en español, comparando LDA y NMF sobre el mismo corpus (objetivo general del informe).
- **Técnica:** Los objetivos específicos del informe (2.2) se cumplen así: delimitación de corpus (módulo carga), preprocesamiento y representaciones adecuadas al español (preproceso y representación), contraste NMF-TF-IDF vs LDA-conteos (modelos), y evaluación con diversidad + coherencia (evaluación).
- **Ejemplo:** Nuestro pipeline completo ejecuta esos 4 objetivos en cadena con `ejecutar_pipeline()`.
- **Repregunta:** «¿Objetivo cumplido?» → El prototipo demuestra el ciclo completo con corpus sintético; el corpus real de 200–400 noticias es la etapa siguiente del proyecto.

**4. ¿Por qué eligieron este problema?**
- **Corta:** Porque las hemerotecas digitales crecen continuamente y revisar cada noticia a mano no escala; además el español boliviano tiene particularidades (tildes, nombres propios, expresiones locales) que hacen interesante el problema de PLN.
- **Técnica:** Es un problema no supervisado genuino: no existen etiquetas temáticas previas para estos asuntos específicos, que además cambian con la coyuntura (sequía, abastecimiento), imposible de cubrir con clases fijas.
- **Ejemplo:** El tema «combustible/abastecimiento» de nuestro corpus atraviesa noticias etiquetadas como transporte, economía o agro en un medio real.
- **Repregunta:** «¿Por qué noticias bolivianas y no otro corpus?» → Por relevancia local y por el reto idiomático: la mayoría de los recursos PLN están pensados para inglés.

**5. ¿Por qué consideran que su sistema es un agente inteligente?**
- **Corta:** Porque recibe información (noticias), actúa para lograr un objetivo (descubrir temas) mediante aprendizaje: no le programamos las respuestas, el modelo las estima de los datos (Russell y Norvig, informe 3.1).
- **Técnica:** Sigue el ciclo de agente: percepción (entrada de documentos) → procesamiento (preprocesamiento, representación, entrenamiento) → salida (temas, palabras, noticias representativas) → respuesta a la solicitud del usuario.
- **Ejemplo:** Si mañana cargamos otro CSV, el mismo agente descubre otros temas sin cambiar una línea de código: se adapta a los datos.
- **Repregunta:** «¿Es un agente como ChatGPT?» → No: es un agente ANALÍTICO, no generativo; no produce texto nuevo, organiza y descubre estructura en documentos.

**6. ¿Qué hace exactamente el agente?**
- **Corta:** Seis etapas: carga y valida noticias, preprocesa los textos en español, los representa como conteos y TF-IDF, entrena LDA y NMF, evalúa los temas (diversidad y coherencia NPMI) y presenta todo en una interfaz Streamlit.
- **Técnica:** Cada etapa vive en un módulo independiente (`src/carga_datos.py`, `preproceso.py`, `representacion.py`, `modelos.py`, `evaluacion.py`) y `src/pipeline.py` las orquesta con `ejecutar_pipeline()`.
- **Ejemplo:** Al presionar «Entrenar LDA» en la app se ejecuta esa cadena completa y en segundos tienes 6 temas con 10 palabras cada uno.
- **Repregunta:** «¿Cuánto tarda?» → Con 30 noticias, ~2–3 segundos; con el corpus real de 400 serán minutos, no horas, porque son matrices pequeñas.

**7. ¿Qué diferencia existe entre su agente y un programa tradicional?**
- **Corta:** Un programa tradicional sigue reglas fijas escritas por el programador; nuestro agente aprende patrones que nadie programó: los temas no están en el código, los estima de las palabras.
- **Técnica:** En un programa tradicional habría que escribir "si contiene 'combustible' → tema combustibles"; nosotros solo definimos el PROCEDIMIENTO (etapas y parámetros) y el contenido emerge del ajuste estadístico (informe 3.2).
- **Ejemplo:** El tema 6 de LDA («salud, caso, semana, reportar») no aparece en ninguna línea de nuestro código: salió del entrenamiento.
- **Repregunta:** «¿Entonces ustedes no influyen en el resultado?» → Sí influyen en la CALIDAD: el preprocesamiento, el número de temas y la representación condicionan qué patrones puede encontrar el modelo.

**8. ¿Qué recibe el agente y qué resultados devuelve?**
- **Corta:** Recibe un CSV (título, medio, fecha, URL, contenido) o archivos TXT; devuelve: temas con sus 10 palabras principales, el peso de cada palabra, las 3 noticias más representativas de cada tema, la distribución de documentos por tema y métricas de diversidad y coherencia.
- **Técnica:** `ejecutar_pipeline(df, n_temas)` devuelve un diccionario con textos procesados, ambas matrices, los dos modelos entrenados y los resúmenes de evaluación listos para la interfaz.
- **Ejemplo:** Para el corpus demo: 6 temas por modelo; NMF con diversidad 1.000 y coherencia 0.368; LDA con diversidad 0.900 y coherencia 0.130.
- **Repregunta:** «¿Devuelve un clasificador?» → No exactamente: devuelve PERTENENCIAS temáticas (mezclas en LDA), un recurso de exploración, no etiquetas definitivas por noticia (informe 4.2).

---

## BLOQUE B — Modelos y técnicas de IA

**9. ¿Qué modelo de IA utilizaron?**
- **Corta:** Dos modelos de topic modeling: LDA (Latent Dirichlet Allocation) y NMF (Non-negative Matrix Factorization), ambos implementados con scikit-learn.
- **Técnica:** LDA (`LatentDirichletAllocation`, src/modelos.py:48) es un modelo probabilístico generativo; NMF (`NMF`, src/modelos.py:65) es un modelo algebraico de factorización de matrices. Ambos aprenden sin etiquetas.
- **Ejemplo:** Entrenamos los dos sobre las mismas 30 noticias y comparamos quién da temas más interpretables.
- **Repregunta:** «¿Cuál es mejor?» → No lo decidimos de antemano (informe 4.1); en nuestro corpus NMF separa más nítido (diversidad 1.000) pero LDA ilustra las mezclas: por eso se comparan.

**10. ¿Qué técnica utilizaron y por qué?**
- **Corta:** La técnica es el modelado temático (topic modeling) dentro del PLN, con aprendizaje no supervisado. La elegimos porque el problema es descubrir temas que no conocemos de antemano.
- **Técnica:** Otras técnicas descartadas y por qué: clasificación supervisada (requiere etiquetas que no existen); clustering como k-means (una noticia queda en UN solo grupo, no puede estar en dos temas — informe 3.8); BERTopic (necesita embeddings y más recursos: queda como ampliación, informe 3.12).
- **Ejemplo:** En nuestro corpus una noticia de salud apareció mezclada en un tema de LDA con combustibles — eso un clasificador no podría representarlo.
- **Repregunta:** «¿Por qué no deep learning?» → Para 30–400 documentos, los modelos léxicos son suficientes, interpretables y no requieren GPU; el informe lo justifica por recursos.

**11. ¿Qué es el aprendizaje no supervisado?**
- **Corta:** Es el aprendizaje donde los datos NO traen la respuesta: el modelo busca estructura (patrones, grupos) sin ejemplos etiquetados. Lo contrario es el supervisado, que aprende de pares entrada→etiqueta.
- **Técnica:** Informe 3.3: en supervisado un clasificador aprende de noticias ya etiquetadas «economía»/«deportes»; en no supervisado solo hay documentos y el modelo estima patrones latentes de co-aparición.
- **Ejemplo:** Nuestro CSV tiene medio y fecha como metadatos, pero NINGUNA columna de tema entra al entrenamiento.
- **Repregunta:** «¿Y entonces para qué sirven los metadatos?» → Para TRAZABILIDAD: mostrar qué noticia concreta representa cada tema, no para entrenar.

**12. ¿Por qué su proyecto es no supervisado y no supervisado?**
- **Corta:** Porque no existen etiquetas previas de los temas específicos que buscamos, y etiquetar miles de noticias a mano sería inviable y sesgado. Queremos descubrir, no verificar clases conocidas.
- **Técnica:** Además los temas de coyuntura (sequía, abastecimiento) no caben en taxonomías fijas de secciones; el no supervisado permite encontrar asuntos MÁS ESPECÍFICOS que las secciones editoriales (informe 3.3).
- **Ejemplo:** El tema de «combustible/diésel/gasolina» de nuestro demo cruza lo que en un medio sería política, economía y local.
- **Repregunta:** «¿Usan las etiquetas para validar?» → Sí para EVALUAR indirectamente: sabemos que el corpus demo tiene 6 temas intencionales y verificamos que el modelo los recupera; eso es prueba, no entrenamiento.

**13. ¿Qué es Topic Modeling?**
- **Corta:** Es una familia de técnicas no supervisadas que descubre estructuras temáticas latentes en una colección de textos, describiendo cada tema con sus palabras características y sus documentos asociados.
- **Técnica:** "Latente" significa que el tema no se observa: se infiere de señales observables (conteos o pesos TF-IDF). Informe 3.9.
- **Ejemplo:** El modelo nunca vio la palabra "combustibles" como etiqueta; la lista «combustible, abastecimiento, gasolina…» emergió de que esas palabras aparecen juntas en las mismas noticias.
- **Repregunta:** «¿Temas = clusters?» → Parecido pero con mezcla: los temas no son mutuamente excluyentes; un documento puede tener 40% de un tema y 30% de otro.

**14. ¿Qué es LDA y cómo funciona?**
- **Corta:** LDA es un modelo probabilístico generativo: supone que cada documento es una MEZCLA de temas y cada tema una DISTRIBUCIÓN de probabilidad sobre las palabras. Entrena invirtiendo ese proceso: observando las palabras, estima las mezclas y las distribuciones.
- **Técnica:** Generativamente: para cada palabra de una noticia, primero se elige un tema según la mezcla del documento y luego una palabra según la distribución de ese tema (Blei et al. 2003, informe 3.10). El ajuste en scikit-learn usa estimación variacional (coordenadas por lotes: `learning_method="batch"`, 30 iteraciones, src/modelos.py:48). Consume CONTEOS de palabras.
- **Ejemplo:** LDA asignó a nuestra noticia 1: 67.7% tema combustibles y 30.2% otro tema — una mezcla, no una etiqueta única.
- **Repregunta:** «¿Qué significa que las proporciones sumen 1?» → Que tienen interpretación probabilística: la mezcla es una distribución sobre temas; lo verificamos en las pruebas con tolerancia 1e-6.

**15. ¿Qué es NMF y cómo funciona?**
- **Corta:** NMF aproxima la matriz TF-IDF (documentos × palabras) como el producto de dos matrices NO negativas más pequeñas: una dice cuánto de cada componente (tema) tiene cada documento, la otra qué peso tiene cada palabra en cada componente.
- **Técnica:** La no-negatividad es la clave: las contribuciones solo se SUMAN (no se restan), lo que produce partes aditivas interpretables (Kuang et al. 2015, informe 3.11). scikit-learn lo resuelve con descenso por coordenadas (`init="nndsvda"`, 500 iteraciones, src/modelos.py:65). Consume TF-IDF.
- **Ejemplo:** NMF separó los 6 grupos de nuestro corpus con nitidez total: 5 noticias exactas por tema, 0 mezclas grandes.
- **Repregunta:** «¿Por qué nndsvda?» → Es una inicialización determinista basada en SVD truncado: más estable que azar puro y reproducible con la semilla.

**16. ¿Cuál es la diferencia entre LDA y NMF?**
- **Corta:** LDA es probabilístico (sus salidas son distribuciones que suman 1 por documento) y consume conteos; NMF es algebraico (sus salidas son coeficientes no negativos que NO son probabilidades) y consume TF-IDF.
- **Técnica:** Informe 3.11 lo dice exacto: "los pesos son coeficientes de la factorización, mientras que las proporciones de LDA tienen interpretación probabilística bajo su modelo". NMF suele separar mejor con vocabularios limpios; LDA representa mezclas de forma natural.
- **Ejemplo:** Misma noticia: LDA 67.7%/30.2%/… (suma 1); NMF 0.299 y casi ceros (no suma 1, son pesos crudos). En grupos: LDA mezcló salud+combustibles en un tema; NMF los separó.
- **Repregunta:** «¿Pueden compararse sus salidas directamente?» → Solo con medidas homogéneas sobre los TEMAS (nuestra NPMI y diversidad), no comparando los números crudos de pertenencia.

**17. ¿Por qué utilizar dos modelos en lugar de uno?**
- **Corta:** Porque el informe exige contrastar dos rutas sobre el mismo corpus y no presuponer cuál es mejor: la comparación es parte del método científico del proyecto (objetivo específico 3 del informe).
- **Técnica:** Comparten corpus preprocesado y vocabulario comparable (mismos parámetros en ambos vectorizadores, src/representacion.py) para que la única diferencia sea el modelo, no los datos.
- **Ejemplo:** En el demo, NMF separó nítido y LDA mezcló: esa ES la respuesta del experimento, con métricas (1.000 vs 0.900 diversidad).
- **Repregunta:** «¿Y si dan igual?» → También sería un resultado: indicaría que la señal temática es robusta a la representación.

**18. ¿Qué es TF-IDF y para qué sirve?**
- **Corta:** Es una ponderación que multiplica la frecuencia de la palabra en el documento (TF) por su rareza en la colección (IDF). Así una palabra común en casi todas las noticias vale poco, y una palabra frecuente pero concentrada en pocas noticias vale mucho.
- **Técnica:** Informe 3.7 (Manning et al. 2008): reduce la influencia de palabras presentes en gran parte de la colección. Es la ENTRADA de NMF. scikit-learn lo implementa en `TfidfVectorizer` con normalización L2 por documento.
- **Ejemplo:** En nuestro demo, «combustible» (aparece solo en 5 noticias, varias veces ahí) pesa más que «nacional» (aparece en muchas noticias distintas).
- **Repregunta:** «¿Y LDA por qué no usa TF-IDF?» → Porque su modelo probabilístico ya modela la frecuencia con distribuciones temáticas; darle TF-IDF rompe su suposición de conteos multinomiales.

**19. ¿Qué es CountVectorizer?**
- **Corta:** Es el vectorizador que construye la matriz documento-término con CONTEOS crudos: en cada celda pone cuántas veces aparece esa palabra en esa noticia. Es la entrada de LDA.
- **Técnica:** Bolsa de palabras (bag of words): ignora el orden. Aplicamos los mismos parámetros de vocabulario que en TF-IDF (`min_df=2`, `max_df=0.9`, `max_features=3000`, src/representacion.py:23) para que ambas rutas trabajen con vocabularios comparables (informe 4.4).
- **Ejemplo:** Nuestra noticia 1 tiene «combustible» 3 veces → en la matriz, su celda vale 3.
- **Repregunta:** «¿Qué hace min_df=2?» → Descarta palabras que aparecen en menos de 2 documentos: son ruido individual que no puede formar tema.

**20. ¿Cómo descubre temas un modelo sin conocer previamente sus nombres?**
- **Corta:** Por estadística de co-aparición: las palabras que tienden a aparecer en los mismos documentos terminan agrupadas. El "nombre" del tema lo pone el humano DESPUÉS, leyendo las palabras y noticias.
- **Técnica:** LDA ajusta sus distribuciones para maximizar la verosimilitud de los conteos observados; los temas que emergen son las fuentes de variación que mejor explican qué palabras aparecen juntas. Informe 3.9: la lista «sequía, represa, riego» sugiere "disponibilidad de agua", y el equipo PROPONE ese nombre revisando documentos.
- **Ejemplo:** El modelo entregó «partido, fútbol, torneo, club, gol» y NOSOTROS lo llamamos tema de deportes; en la UI, cada tema se propone como etiqueta legible (`etiqueta_tema`).
- **Repregunta:** «¿Puede el modelo juntar palabras sin sentido común?» → Sí, pasa: por eso la coherencia NPMI y la revisión humana son obligatorias; un grupo de palabras puede co-aparecer por estilo editorial y no por tema (informe 3.13).

---

## BLOQUE C — Implementación y desarrollo

**21. ¿Cómo desarrollaron el agente paso a paso?**
- **Corta:** Con desarrollo guiado por pruebas (TDD): definimos la arquitectura de 6 módulos, y para cada módulo escribimos primero la prueba que debía pasar, la vimos fallar, y después implementamos. Terminamos con 59 pruebas verdes y una revisión independiente.
- **Técnica:** El orden fue: corpus demo → carga → preprocesamiento → representaciones → modelos → evaluación → pipeline → interfaz → documentación. Cada módulo tiene su archivo de pruebas con el mismo nombre.
- **Ejemplo:** La prueba de tildes (`tests/test_preproceso.py:35`) exigía que «público» y «publicó» dieran términos distintos ANTES de escribir el preprocesador.
- **Repregunta:** «¿Cómo garantizan que no hay resultados falsos?» → Hay una prueba de determinismo (`tests/test_integracion.py`): dos ejecuciones deben dar EXACTAMENTE los mismos temas; y todo número en pantalla sale de un modelo entrenado en esa sesión.

**22. ¿Qué lenguaje utilizaron y por qué?**
- **Corta:** Python, porque es el estándar en IA/PLN y concentra el ecosistema necesario (scikit-learn, spaCy, pandas) con interfaces simples y gratuitas.
- **Técnica:** Python 3.11+; todo corre local sin GPU ni servicios pagos, como exigía el alcance (informe 4.4).
- **Ejemplo:** El pipeline completo cabe en ~6 archivos cortos gracias a las abstracciones de scikit-learn.
- **Repregunta:** «¿Limitaciones de Python?» → Rendimiento numérico puro, mitigado porque numpy/sklearn ejecutan en C; y el entorno virtual como requisito de instalación.

**23. ¿Qué bibliotecas utilizaron?**
- **Corta:** scikit-learn (LDA, NMF, vectorizadores), spaCy con `es_core_news_sm` (lematización y stopwords del español), pandas (datos), NumPy (numérica), matplotlib (gráficos), Streamlit (interfaz) y pytest (pruebas).
- **Técnica:** Separar responsabilidades: spaCy NO participa del entrenamiento (solo prepara texto); scikit-learn NO sabe de interfaz; Streamlit solo presenta. El informe (3.6) pedía justamente bibliotecas adecuadas para español.
- **Ejemplo:** El lema «exportación» que muestra la página de preprocesamiento lo produce spaCy; el tema que agrupa esa palabra lo produce scikit-learn.
- **Repregunta:** «¿Por qué no gensim?» → Lo evaluamos y descartamos: es frágil de instalar en Windows con Python nuevo y solo lo necesitábamos para coherencia; implementamos NPMI nosotros con NumPy (misma medida para ambos modelos).

**24. ¿Cómo obtuvieron o prepararon los documentos?**
- **Corta:** Para el prototipo generamos un corpus SINTÉTICO de 30 noticias con un script determinista (semilla 2026): 6 temas × 5 noticias, medios ficticios "Medio Demo 1/2", declarado claramente como demostrativo. El corpus real (200–400 noticias de hasta 2 medios, con autorización) es la siguiente etapa (informe 4.3).
- **Técnica:** El generador (`datos/generar_noticias_demo.py`) mezcla bancos de frases temáticas + conectores editoriales compartidos: suficiente señal para que los temas sean descubribles pero no triviales.
- **Ejemplo:** `datos/LEEME.txt` y el badge de la app advierten: "ningún texto proviene de un medio real".
- **Repregunta:** «¿Por qué no rasparon noticias reales ya?» → Por las condiciones de uso: el informe exige verificar disponibilidad y permiso antes de extraer; hacerlo sin permiso sería ilegal/ético-mente incorrecto.

**25. ¿Qué es el preprocesamiento de texto?**
- **Corta:** Es preparar el texto crudo para que el modelo solo reciba señal temática útil: minúsculas, quitar URLs/signos/números, tokenizar, eliminar palabras vacías y lematizar («subieron»→«subir»).
- **Técnica:** Informe 3.6: también detectar republicaciones y revisar qué elimina cada regla. Nuestro pipeline: `limpiar_texto` → `tokenizar` → filtro spaCy (alfabético, ≥3 letras, no stopword) → lemas (`src/preproceso.py:69`).
- **Ejemplo:** «El transporte pesado denunció retrasos…» se convierte en «transporte pesado denunciar retraso abastecimiento combustible cochabamba…».
- **Repregunta:** «¿Puede quitar preprocesamiento de más?» → Sí: sobre-limpieza destruye señal. Por eso NO stemming agresivo y NO quitar tildes: «público» y «publicó» serían la misma palabra.

**26. ¿Cómo convierten las palabras en números?**
- **Corta:** Con dos matrices documento-término sobre el mismo vocabulario: CONTEOS (CountVectorizer) para LDA y TF-IDF (TfidfVectorizer) para NMF. Cada noticia queda como un vector: una fila con un número por palabra del vocabulario.
- **Técnica:** Parámetros idénticos en ambos (`min_df=2, max_df=0.9, max_features=3000`) para vocabularios comparables; `max_df=0.9` poda términos presentes en >90% de noticias porque no distinguen temas (informe 3.7).
- **Ejemplo:** La noticia 1 se vuelve un vector donde la posición de «combustible» vale 3 (conteos) o su peso ponderado (TF-IDF).
- **Repregunta:** «¿Y el orden de las palabras?» → Se pierde: es la hipótesis de bolsa de palabras; para temas funciona porque lo que importa es QUÉ palabras co-aparecen, no en qué orden.

**27. ¿Cómo se entrenan LDA y NMF?**
- **Corta:** LDA optimiza variacionalmente sus distribuciones tema-palabra y documento-tema para explicar los conteos (30 iteraciones, batch); NMF minimiza el error de reconstrucción de la matriz TF-IDF con descenso por coordenadas (hasta 500 iteraciones), imponiendo no-negatividad.
- **Técnica:** Ambos con `random_state=42` y configuración determinista (batch / nndsvda) para reproducibilidad (informe 4.4 exige semillas documentadas). Validamos antes de entrenar: nº documentos ≥ nº temas y matriz no vacía (`CorpusInsuficienteError`).
- **Ejemplo:** La prueba de reproducibilidad (`tests/test_modelos.py`) entrena dos veces y exige las mismas palabras por tema.
- **Repregunta:** «¿Qué es el random_state?» → La semilla del generador aleatorio: fijarla hace que dos ejecuciones den idéntico resultado, esencial para poder citar resultados en la defensa.

**28. ¿Cómo funciona la interfaz del agente?**
- **Corta:** Una aplicación Streamlit con 6 páginas en la barra lateral: Inicio, Cargar noticias, Preprocesamiento, Modelo LDA, Modelo NMF y Comparación. El estado (documentos cargados, textos procesados, pipeline entrenado) se guarda en `st.session_state` entre páginas.
- **Técnica:** Streamlit re-ejecuta el script en cada interacción; por eso el estado persistente va en session_state y los cálculos pesados se disparan solo con botones (app.py:167, 241, 298). Si cambias el corpus, se limpian las etapas dependientes (app.py:184).
- **Ejemplo:** En la página LDA eliges 2–10 temas con un slider (app.py:295), entrenas y ves la tabla de temas, el gráfico de pesos por tema y las noticias representativas con su medio, fecha y enlace.
- **Repregunta:** «¿Por qué entrenar entrena LDA Y NMF?» → Para que la comparación (página 5) sea siempre sobre el mismo corpus entrenado en la misma sesión; lo anuncia el propio mensaje de éxito.

**29. ¿Qué sucede internamente cuando el usuario presiona el botón de analizar?**
- **Corta:** Se llama a `ejecutar_pipeline(df, n_temas)`, que en cadena: preprocesa los 30 textos con spaCy, construye la matriz de conteos y la de TF-IDF, entrena LDA y NMF con semilla 42, calcula palabras por tema, pesos por documento, distribución predominante, diversidad y coherencia NPMI, y guarda todo en el estado de sesión.
- **Técnica:** Detalle por módulo: `procesar_corpus` (preproceso.py:84) → `construir_matriz_conteos`/`construir_matriz_tfidf` (representacion.py:49,54) → `entrenar_lda`/`entrenar_nmf` (modelos.py:48,65) → `resumen_modelo` por modelo (evaluacion.py:116). La interfaz luego solo FORMATEA (app.py:41 `tabla_temas`).
- **Ejemplo:** En el demo eso significa: 30 documentos → vocabulario de ~170 términos útiles → 2 modelos → 12 listas de palabras + 60 noticias representativas en ~2–3 segundos.
- **Repregunta:** «¿Y si algo falla ahí?» → Los errores de dominio (corpus vacío, vocabulario inutilizable, más temas que documentos) llegan a la UI como mensajes accionables en español vía `st.error`, nunca como traceback.

**30. ¿Qué dificultades técnicas encontraron o podrían encontrar?**
- **Corta:** Encontramos y resolvimos: cómo preservar tildes sin fusionar palabras, cómo hacer vocabularios comparables entre las dos rutas, y cómo implementar coherencia NPMI sin dependencias frágiles. Podremos encontrar: vocabulario dominante de un medio que distorsione temas, y rendimiento con 400 noticias.
- **Técnica:** El informe (4.5) anticipa exactamente ese riesgo de medio dominante; nuestra mitigación es el `max_df` y, en el corpus real, comparar temas por medio. La co-ocurrencia para NPMI se recalcula por llamada: con 400 noticias convendrá computarla una vez (ya identificado en la revisión).
- **Ejemplo:** La prueba del corpus "solo stopwords" (`tests/test_representacion.py`) surgió de preguntarnos qué pasa si el preprocesamiento deja textos vacíos: la respuesta es un error accionable, no un crash.
- **Repregunta:** «¿El mayor desafío del proyecto?» → Interpretar los temas: la métrica orienta, pero decidir que «combustible, diésel, abastecimiento» ES "abastecimiento de combustibles" requiere leer noticias (informe 4.5).

---

## BLOQUE D — Resultados y validación

**31. ¿Qué resultados entrega el agente?**
- **Corta:** Para cada modelo: 6 temas con sus 10 palabras y su peso, la coherencia NPMI por tema, las 3 noticias más representativas por tema, la distribución de noticias por tema predominante, y por modelo: diversidad y coherencia media.
- **Técnica:** Todo sale de `resumen_modelo` (evaluacion.py:116) y se presenta en `tabla_temas` + gráficos. Números reales del demo: NMF diversidad 1.000, coherencia 0.368; LDA diversidad 0.900, coherencia 0.130.
- **Ejemplo:** Tema 1 de NMF: «combustible, abastecimiento, gasolina, transporte, diésel, frontera…» con 5 noticias predominantes.
- **Repregunta:** «¿El agente "acierta"?» → En el demo recupera los 6 temas intencionales con NMF; con LDA aparecen mezclas (salud+combustibles): ambas observaciones son resultados, no fallos del prototipo.

**32. ¿Cómo saben si los temas descubiertos son correctos o coherentes?**
- **Corta:** Con dos medidas homogéneas y con revisión humana: diversidad (proporción de términos únicos entre temas) y coherencia NPMI (si las palabras del tema realmente co-aparecen), más la lectura de noticias representativas para nombrar cada tema.
- **Técnica:** Informe 3.13: la comparación exige misma medida, mismo corpus y mismo número de términos para ambos modelos; y las métricas SE COMPLEMENTAN con revisión humana (Chang et al. 2009: mejor score no garantiza interpretabilidad). NPMI = log(p(w1,w2)/(p(w1)p(w2))) / −log(p(w1,w2)), con ventanas deslizantes de 10 tokens sobre el propio corpus.
- **Ejemplo:** El tema mejor coherente de NMF (salud, 0.543) tiene palabras que sí conviven; el tema 2 de LDA mezcló salud con combustibles y su coherencia es negativa (−0.179): la métrica lo señala.
- **Repregunta:** «¿NPMI negativa es tema malo?» → Indica palabras que NO co-aparecen más que por azar: sospecha de tema fragmentado o mezcla; se confirma leyendo las noticias.

**33. ¿Cómo comparan los resultados de LDA y NMF?**
- **Corta:** Con la misma medida para ambos (NPMI y diversidad sobre sus listas de palabras), la distribución de documentos por tema lado a lado, y la interpretabilidad: qué palabras y qué noticias produce cada uno sobre el MISMO corpus y vocabulario.
- **Técnica:** La página 5 de la app muestra métricas, tablas y gráficos juntos; el criterio del informe (4.5) es cuál ofrece temas más comprensibles en los documentos seleccionados, no cuál "gana" por un número.
- **Ejemplo:** NMF: 6 grupos limpios de 5 noticias (perfectamente balanceados); LDA: 8/4/5/5/5/3 con un tema mezcla — el contraste de filosofías se ve en los datos.
- **Repregunta:** «¿Conclusión de la comparación?» → En corpus pequeños con vocabulario limpio, NMF separa mejor; LDA aporta la visión de mezclas. Con el corpus real puede cambiar, y eso es lo que se investigará.

**34. ¿Qué sucede cuando una noticia habla de varios temas?**
- **Corta:** En LDA recibe proporciones en varios temas (mezcla probabilística); en NMF puede tener peso en varios componentes, aunque con pesos crudos. La UI muestra solo el tema PREDOMINANTE en la distribución, pero los pesos completos están accesibles.
- **Técnica:** Es la ventaja conceptual del topic modeling sobre el clustering (informe 3.8): la noticia de "escasez de diésel y cosechas" no necesita elegir entre combustibles y agricultura.
- **Ejemplo:** Nuestra noticia 1 (abastecimiento, pero con vocabulario de autoridades) recibió en LDA 67.7% combustibles y 30.2% salud/gobierno: el modelo captó el carácter mixto.
- **Repregunta:** «¿Cómo la clasifican entonces?» → No la clasificamos: le asignamos su tema predominante PARA ORDENAR la vista, dejando claro en la interfaz que es una mezcla.

**35. ¿Qué limitaciones tiene el prototipo?**
- **Corta:** Corpus sintético y pequeño (30 noticias), sin embargo claramente marcado; resultados dependientes de ese corpus; bolsa de palabras (pierde orden y negaciones); interpretación de temas sujeta a revisión humana; y falta el corpus real, el barrido de número de temas y BERTopic.
- **Técnica:** Informe 4.5: los temas describen la MUESTRA, no "la prensa boliviana"; y un medio con vocabulario dominante puede sesgar. La bolsa de palabras no capta "no aumentó" ≠ "aumentó".
- **Ejemplo:** Si cargamos 30 noticias de un solo tema, los 6 "temas" serían fragmentos de ese tema: la calidad depende de la colección.
- **Repregunta:** «¿Es grave usar datos sintéticos?» → Para un prototipo metodológico, no: sirve para validar el PIPELINE completo; lo crucial es no presentar esos números como resultados sobre prensa real, y el proyecto lo declara en todas partes.

---

## BLOQUE E — Preguntas difíciles de la ingeniera

**36. Si el modelo no tiene etiquetas, ¿cómo sabe que un tema corresponde a economía o deportes?**
- **Corta:** NO lo sabe y no lo necesita: el modelo solo sabe que esas palabras co-aparecen; la correspondencia con "economía" o "deportes" la establece el lector humano al interpretar la lista.
- **Técnica:** Los temas son latentes y sin nombre (informe 3.9). La etiqueta humana es una POST-INTERPRETACIÓN para comunicar resultados, no parte del modelo.
- **Ejemplo:** Nosotros leímos «partido, fútbol, torneo, club, gol» y lo llamamos "deportes"; otra persona podría llamarlo "fútbol local" y ambas etiquetas describen el mismo patrón.
- **Repregunta:** «¿Y si dos personas lo nombran distinto?» → Es esperado e incluso útil: el informe propone registro de desacuerdos entre revisores como parte de la validación (3.13).

**37. ¿Qué pasaría si cambian el número de temas?**
- **Corta:** Los temas se re-particionan: con muy pocos fusiona asuntos distintos; con demasiados fragmenta temas cercanos o crea temas de "estilo". El slider permite ensayar 2–10 y la coherencia NPMI ayuda a elegir.
- **Técnica:** El número de componentes es un hiperparámetro del modelo (informe 3.10 y 4.4): se debe ensayar una serie de valores, no asumir uno. Con corpus pequeño hay piso: no puede haber más temas que documentos (`CorpusInsuficienteError`).
- **Ejemplo:** Con n_temas=3 en nuestro demo, combustibles y economía probablemente se fusionarían en un tema "economía general" (ambos comparten vocabulario de precios).
- **Repregunta:** «¿Cómo elegirían el definitivo?» → Barrido de k con coherencia + diversidad + interpretabilidad, documentando la semilla de cada corrida (planificado para el corpus real).

**38. ¿Qué diferencia existe entre descubrir temas y clasificar documentos?**
- **Corta:** Clasificar asigna documentos a CLASES CONOCIDAS de antemano (supervisado, necesita ejemplos etiquetados); descubrir temas encuentra las clases mismas que ni sabías que existían (no supervisado), y permite pertenencia múltiple.
- **Técnica:** El clasificador responde "¿qué ES esta noticia?"; el topic model responde "¿DE QUÉ está hecha esta colección?". Informe 3.3 contraste directo.
- **Ejemplo:** Un clasificador entrenado con secciones nunca inventaría "abastecimiento de combustibles"; nuestro modelo lo sacó de los datos.
- **Repregunta:** «¿Se puede combinar ambos?» → Sí: descubrir temas primero y luego entrenar un clasificador con esos temas como etiquetas; es una extensión natural del proyecto.

**39. ¿Qué aporta realmente la IA a su solución?**
- **Corta:** El aprendizaje automático: la capacidad de ESTIMAR la estructura temática de los datos en lugar de definirla. Sin IA, esto sería leer 400 noticias y agruparlas a mano.
- **Técnica:** La IA (Russell y Norvig, informe 3.1) aporta el modelo generativo/factorización que convierte conteos en temas, y la estadística (NPMI) para valorarlos. El resto es ingeniería de software que PREPARA los datos y PRESENTA resultados.
- **Ejemplo:** Cambiar el corpus y re-ejecutar produce un mapa temático nuevo sin reprogramar nada: eso es aprendizaje, no lógica fija.
- **Repregunta:** «¿Y la parte "inteligente" es realmente inteligente?» → Es aprendizaje estadístico, no comprensión del lenguaje: no sabe qué es "diésel", solo que viaja con "gasolina". Es honesto decirlo y es exactamente por qué la revisión humana es parte del diseño.

**40. ¿Qué tienen implementado actualmente y qué falta para terminar el proyecto?**
- **Corta:** Implementado: pipeline completo real (carga, preprocesamiento español, representaciones comparables, LDA, NMF, evaluación NPMI/diversidad, interfaz, 59 pruebas, corpus sintético 30 noticias, documentación). Falta: corpus real 200–400 noticias, barrido del número de temas, coherencia Cv sobre corpus de referencia, protocolo de revisión humana a dos ciegas y BERTopic como ampliación.
- **Técnica:** Informe 4.4–4.5 y 3.12–3.13 definen ese roadmap; nada de lo mostrado hoy fue simulado: cada número sale de modelos entrenados en vivo y es reproducible con semilla 42.
- **Ejemplo:** "Hoy entrenamos esto EN VIVO ante usted" es la mejor demostración: la app entrena en segundos ante el docente.
- **Repregunta:** «¿Cuánto falta?» → El pipeline está; la fase restante es principalmente ADQUISICIÓN de corpus y protocolo de evaluación, no desarrollo nuevo de código.

---

## 5. EXPLICACIÓN PASO A PASO del funcionamiento (para explicar en pizarra)

Con la noticia 1 del corpus como hilo conductor (números reales de ejecución):

1. **Entrada** — El usuario sube un CSV o usa el corpus demo (30 noticias, medios "Medio Demo 1/2").
2. **Lectura y validación** (`src/carga_datos.py:63`) — Se normalizan encabezados (acepta "Título"/"URL"), se descartan noticias con menos de 5 palabras y duplicados por contenido/URL. Informe visible: leídas/inválidas/duplicadas/válidas.
3. **Limpieza y preprocesamiento** (`src/preproceso.py:48,69`) — Minúsculas, sin URLs/signos; spaCy lematiza y quita stopwords. «El transporte pesado denunció retrasos en el abastecimiento de combustible…» → «transporte pesado denunciar retraso abastecimiento combustible cochabamba subsidio…». Tildes PRESERVADAS.
4. **Representación numérica** (`src/representacion.py:49,54`) — Dos matrices del mismo corpus y vocabulario: conteos enteros (LDA) y pesos TF-IDF entre 0 y 1 (NMF).
5. **Entrenamiento LDA** (`src/modelos.py:48`) — Variacional batch, 30 iteraciones, semilla 42. Salida: 6 distribuciones tema-palabra y una mezcla por noticia. Noticia 1 → 67.7% T1 (combustibles), 30.2% T2.
6. **Entrenamiento NMF** (`src/modelos.py:65`) — Factorización TF-IDF no negativa (nndsvda, CD). Salida: componentes y coeficientes crudos. Noticia 1 → 0.299 en T1, ~0 en el resto.
7. **Palabras representativas** (`src/modelos.py:82`) — Las 10 palabras con mayor peso de cada tema, usando `components_` (mismo extractor para ambos modelos).
8. **Identificación de temas por documento** (`src/modelos.py:91,112,117`) — Pesos por documento, tema predominante (argmax) y las 3 noticias más representativas por tema (top pesos del tema).
9. **Comparación y evaluación** (`src/evaluacion.py:27,84,116`) — Diversidad (términos únicos) y NPMI (co-aparición en ventanas deslizantes de 10 tokens, pares no ordenados, suavizado 1e-12), LA MISMA medida para ambos. Resultado demo: NMF 1.000/0.368 vs LDA 0.900/0.130.
10. **Presentación** (`app.py:41,71,158–437`) — Tablas, gráficos de barras (colores fijos por modelo) y noticias con enlace; páginas de comparación lado a lado.

---

## 6. MODELO vs TÉCNICA vs ALGORITMO (no confundir)

| Pregunta | Respuesta | No confundir con |
|---|---|---|
| **Área de IA** | Inteligencia Artificial → Aprendizaje Automático → **Aprendizaje no supervisado** | El área no es "PLN" (eso es la subárea de aplicación) |
| **Subárea** | **Procesamiento de Lenguaje Natural (PLN)** | PLN no es un algoritmo: es el campo que aporta las técnicas textuales |
| **Técnica** | **Modelado temático (topic modeling)**, exploratorio, sobre documentos | La técnica NO es "TF-IDF" (eso es una representación) |
| **Modelos** | **LDA** (probabilístico generativo) y **NMF** (factorización matricial no negativa) | "Modelo" no es "biblioteca": scikit-learn no es un modelo |
| **Algoritmos de ajuste** | LDA: **estimación variacional** (batch); NMF: **descenso por coordenadas** con inicialización NNDSVDa | El algoritmo es CÓMO se optimiza el modelo, no el modelo mismo |
| **CountVectorizer / TF-IDF** | **Representación numérica** del texto (preprocesamiento de características) | No son modelos ni aprenden: solo transforman texto en matrices |
| **spaCy** | Biblioteca de PLN que aporta **lematización y stopwords** del español | No entrena nuestros modelos; solo prepara la entrada |
| **scikit-learn** | **Biblioteca** que implementa LDA, NMF y vectorizadores | La biblioteca es la herramienta; el modelo es el objeto matemático que ella construye |
| **Streamlit** | Framework de **interfaz web** en Python | No tiene ninguna parte de IA: solo presenta resultados |
| **pandas / NumPy / matplotlib** | Bibliotecas de **datos, numérica y gráficos** | Utilería, no aprendizaje |

Frase de cierre para la defensa: *"Nuestra área es aprendizaje no supervisado, nuestra técnica es topic modeling sobre PLN, nuestros modelos son LDA y NMF ajustados por algoritmos variacionales y de descenso por coordenadas en scikit-learn; TF-IDF y conteos son las representaciones de entrada, y Streamlit es solo la vitrina."*

---

## 7. PREGUNTAS SOBRE EL CÓDIGO (archivo, función, líneas reales)

Dónde está cada cosa (verificadas en el código entregado):

| ¿Dónde…? | Archivo | Función | Línea |
|---|---|---|---|
| Se cargan las noticias (CSV) | src/carga_datos.py | `cargar_csv` | 63 |
| Se cargan TXT | src/carga_datos.py | `cargar_carpeta_txt` | 83 |
| Se validan/descartan vacíos y duplicados | src/carga_datos.py | `_limpiar_dataframe` | 37 |
| Se limpian los textos (minúsculas, URLs, signos) | src/preproceso.py | `limpiar_texto` | 48 |
| Se tokeniza | src/preproceso.py | `tokenizar` | 57 |
| Pipeline completo de preprocesamiento (lemas) | src/preproceso.py | `procesar_texto` / `procesar_corpus` | 69 / 84 |
| Se decide si un documento es válido | src/preproceso.py | `es_documento_valido` | 99 |
| CountVectorizer (entrada LDA) | src/representacion.py | `construir_matriz_conteos` | 49 |
| TF-IDF (entrada NMF) | src/representacion.py | `construir_matriz_tfidf` | 54 |
| Parámetros del vocabulario compartido | src/representacion.py | `PARAMS_VOCABULARIO` | 23 |
| Se entrena LDA | src/modelos.py | `entrenar_lda` | 48 |
| Se entrena NMF | src/modelos.py | `entrenar_nmf` | 65 |
| Palabras por tema | src/modelos.py | `palabras_por_tema` | 82 |
| Pesos documento-tema | src/modelos.py | `pesos_documento` | 91 |
| Diversidad temática | src/evaluacion.py | `diversidad_tematica` | 27 |
| Coherencia NPMI (ventanas) | src/evaluacion.py | `coherencia_npmi` / `_ventanas_por_documento` | 84 / 38 |
| Resumen de evaluación de un modelo | src/evaluacion.py | `resumen_modelo` | 116 |
| Orquestador de las 6 etapas | src/pipeline.py | `ejecutar_pipeline` | 27 |
| Botón de cargar / procesar / entrenar | app.py | `pagina_cargar` / `pagina_preproceso` / `_pagina_modelo` | 167 / 241 / 298 |
| Slider de número de temas | app.py | `_pagina_modelo` | 295 |
| Tabla y etiquetas de temas (presentación) | app.py | `tabla_temas` / `etiqueta_tema` | 41 / 57 |
| Navegación de la app | app.py | `main` | 439 |

Preguntas que la ingeniera podría hacer mirando el código (con respuesta):

- **«¿Por qué dos vectorizadores si el vocabulario debe ser comparable?»** → Porque comparar modelos exige que solo cambie el modelo; por eso ambos comparten `PARAMS_VOCABULARIO` (representacion.py:23) y hay una prueba que exige vocabularios idénticos.
- **«¿Por qué `learning_method="batch"`?»** → Con corpus pequeños el método batch es determinista y estable con semilla; el online está pensado para corpus enormes.
- **«¿Dónde evitarían que un tema "aparezca" sin haberse entrenado?»** → No se puede: los temas se construyen en `fit()`; la UI solo lee `components_` y `transform()`. No hay ningún valor hardcodeado de temas.
- **«¿Esta línea qué hace?» (evaluacion.py:76 `_npmi_par`)** → Calcula la NPMI de un par: PMI normalizado por −log p(w1,w2), con suavizado 1e-12 para pares nunca vistos; queda acotada a (−1, 1).
- **«¿Por qué la semilla está en un módulo y no hardcodeada?»** → `SEMILLA_ALEATORIA=42` (modelos.py) se usa en ambos modelos y se muestra en la UI: reproducibilidad documentada (informe 4.4).

---

## 8. LO PROPUESTO vs LO IMPLEMENTADO vs LO PROBADO

| Informe PROPONE | ¿Estado? | Evidencia |
|---|---|---|
| Corpus 200–400 noticias reales, 2 medios (4.3) | **Pendiente** | Hoy: corpus sintético de 30, marcado como demo (LEEME + badge UI) |
| Preprocesamiento español con tildes cuidadas (3.6) | ✅ Implementado y probado | spaCy; test «público»≠«publicó»; lemas «exportación» |
| Dos rutas: LDA-conteos, NMF-TF-IDF, vocabulario comparable (3.7, 4.4) | ✅ Implementado y probado | `PARAMS_VOCABULARIO` compartido; test de vocabularios idénticos |
| Semillas documentadas y reproducibles (4.4) | ✅ Implementado y probado | `SEMILLA_ALEATORIA=42`; test de determinismo del pipeline completo |
| Evaluación: coherencia + diversidad homogéneas (3.13) | ✅ Implementado (NPMI propia) | `evaluacion.py`; la Cv sobre corpus externo queda para el proyecto final |
| Interpretación humana de temas (3.13, 4.5) | ⏳ Parcial | La app muestra noticias representativas para nombrar temas; el protocolo a dos revisores es del proyecto final |
| Comparación k de número de temas (3.10, 4.4) | ⏳ Parcial | Slider 2–10 en UI; el barrido sistemático con métricas es de la fase real |
| k-means como contraste conceptual (3.8) | ⏳ Discutido, no implementado | Es contraste conceptual del informe, no requisito del prototipo |
| BERTopic (3.12) | ❌ Opcional futuro | Informe: solo si hay codificador español y recursos |

**Lo probado, con evidencia:** 59 pruebas pytest (todas pasan), prueba de determinismo (mismos temas en 2 ejecuciones), revisión independiente de código, y batería de 5 agentes de prueba adicionales. **Regla para la defensa:** nunca decir "resultados sobre prensa boliviana" — decir "resultados sobre el corpus sintético de demostración; el corpus real es la siguiente etapa".

---

## 9. SIMULACRO DE DEFENSA ORAL (cómo practicar conmigo)

Cuando quieras practicar, dime **"empezamos el simulacro"** en el chat y:

1. Asumo el papel de la ingeniera exigente y te hago UNA pregunta a la vez (empiezo por las del Bloque A y voy subiendo a las del E)
2. Tú respondes con tus palabras (como en oral, no copiado)
3. Yo evalúo: nota 1–10, corrección técnica, qué faltó, errores conceptuales, una mejor versión y una repregunta
4. No avanzo a la siguiente hasta que esa quede sólida

Consejo de oral: responde en 3 tiempos — *qué es (1 frase) → cómo funciona en NUESTRO proyecto (1-2 frases) → ejemplo concreto con números* (el 67.7%/30.2%, el 1.000 de diversidad, «combustible, abastecimiento, gasolina…»).

---

## 10. RESUMEN FINAL — las 8 respuestas que debes poder dar dormido

- **¿Qué modelo?** LDA y NMF (scikit-learn), comparados sobre el mismo corpus.
- **¿Qué técnica?** Topic modeling (PLN), exploratorio.
- **¿Por qué no supervisado?** No hay etiquetas previas; se descubren temas, no se clasifican en clases conocidas.
- **¿Cómo lo desarrollaron?** 6 módulos desacoplados con TDD: 59 pruebas; revisión independiente.
- **¿Cómo funciona internamente?** Carga → preproceso (spaCy, tildes preservadas) → conteos y TF-IDF → entrenamiento (semilla 42) → palabras/pesos/noticias → diversidad + NPMI → Streamlit.
- **¿Cómo entrenaron?** LDA variacional batch (30 it.) sobre conteos; NMF CD con nndsvda (500 it.) sobre TF-IDF; semilla 42; validaciones previas.
- **¿Qué resultados?** Demo: NMF 6 temas nítidos (div 1.000, coh 0.368); LDA mezclas (div 0.900, coh 0.130) — ambos reales y reproducibles.
- **¿Cómo demuestran que funciona?** 59 pruebas + determinismo + entrenamiento EN VIVO ante el docente + todo el código y datos en GitHub.
