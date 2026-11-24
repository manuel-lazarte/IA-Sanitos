# BERTopic/HDBSCAN: decisión e integración

Fecha: 24/11/2026. Base inspeccionada: `0cfc1bf`, rama `feature/bertopic-hdbscan`.
El árbol estaba limpio. Python 3.11.9 ya estaba disponible en `.venv`.
Las 59 pruebas originales pasaron antes de modificar el código (27.08 s).

## Fuentes y precedencia

Se revisaron `InformeIASANITOS.pdf` (anexo de tres páginas + informe de 18),
`contextoProyectoIASANITOS.md` (encabezado «Contexto maestro»), el encargo
`t1BERTOPIC-HDBSCAN(implementar).md`, README, documentación de `docs/`,
fuentes y pruebas. No se encontró un archivo previo de decisión técnica ni
el contexto bajo el nombre literal `CONTEXTO_COMPLETO_IA_SANITOS_GRUPO15.md`;
el adjunto aportado cumple esa función.

Contradicciones: el informe original, README y guías previas describían BERTopic
como futuro/opcional. La solicitud vigente y el anexo metodológico establecen
conservar LDA/NMF e incorporar BERTopic/HDBSCAN. Se aplica ese alcance sin
atribuir al PDF una aprobación escrita de la docente. La documentación histórica
de `docs/superpowers` describe el prototipo inicial, no el estado de esta rama.
El informe menciona C_v como ejemplo; el código implementa NPMI.

## Arquitectura y decisiones

`src/bertopic_modelo.py` implementa la tercera ruta de forma independiente.
El pipeline solo importa/ejecuta la pila semántica cuando se solicita BERTopic.
No usa pesos, temas, matrices ni lemas producidos por LDA/NMF. La llamada anterior
al pipeline conserva su formato y comportamiento. Las selecciones parciales
devuelven únicamente los artefactos de los modelos solicitados.

1. **Embeddings:** `intfloat/multilingual-e5-small`, 384 dimensiones, CPU,
   Sentence Transformers, normalización L2, prefijo `query: ` para clustering.
   Se elige por soporte multilingüe y coste menor que las variantes base/large;
   no se afirma que esté validado específicamente para español boliviano.
   No se entrena el codificador ni se envían noticias a una API externa.
2. **Noticias largas:** límite efectivo `min(512, encoder.max_seq_length)`.
   Bisección del texto en fronteras de espacios cuando sea posible, comprobando
   con el tokenizador cada entrada prefijada y sus tokens especiales. Conserva
   todos los caracteres del cuerpo tras normalizar espacios, incluido el final;
   no hay solapamiento. Media de embeddings de fragmentos ponderada por sus
   tokens y normalización final. Registra fragmentos por noticia. Se pierde
   orden entre fragmentos y puede diluirse un asunto minoritario; las fronteras
   no siempre coinciden con oraciones. La tokenización de control usa
   `truncation=False`; los fragmentos que llegan a encode ya caben.
3. **UMAP real:** coseno, hasta 15 vecinos, hasta 5 dimensiones, min_dist=0,
   semilla 42, un hilo e inicialización aleatoria para evitar errores de la
   inicialización espectral en corpus pequeños. Vecinos <= N-1 y dimensiones
   <= N-2. La proyección modifica distancias/densidades y debe evaluarse.
4. **HDBSCAN real:** paquete `hdbscan`, distancia euclídea sobre UMAP,
   min_cluster_size=5, min_samples=2, selección EOM, prediction_data=True,
   un hilo para distancias. No fija K, no fuerza un único grupo y no reasigna
   ruido. Son parámetros iniciales, no optimizados con el corpus definitivo.
5. **c-TF-IDF real:** `ClassTfidfTransformer` de BERTopic sobre textos reunidos
   por grupo. CountVectorizer usa unigramas/bigramas, min_df=1 (se aplica sobre
   grupos), stopwords españolas/editoriales y letras de al menos tres caracteres,
   preservando tildes. BERTopic usa explícitamente `language="multilingual"`
   para evitar la limpieza inglesa interna que elimina letras acentuadas.
   Los bigramas se forman después de quitar stopwords y
   pueden conectar palabras separadas en el texto original; requieren revisión.
   Los embeddings conservan puntuación, mayúsculas y negaciones.

El mínimo operativo es cinco noticias: entradas menores, vacías, no textuales,
sin vocabulario útil o con parámetros incompatibles se rechazan con mensajes
en español. No se eliminan filas dentro de esta ruta. La carga CSV existente
sí valida y elimina duplicados antes, por igual para los tres métodos.

## Contrato para frontend

```python
r = ejecutar_pipeline(df, modelos=("BERTopic",))
modelo = r["bertopic"]                 # objeto ajustado para inspección
s = r["resumenes"]["BERTopic"]
noticias_tema = r["df"].iloc[s["temas"][0]["doc_ids"]] if s["temas"] else None
sin_grupo = r["df"].iloc[s["atipicos"]["doc_ids"]]
```

| Campo del resumen | Significado |
|---|---|
| nombre, n_documentos, n_temas, tema_ids | Identificación y conteos; n_temas excluye -1 |
| temas | Lista de tema, palabras, pesos_ctfidf, doc_ids, representativos, coherencia |
| asignaciones | Un ID temático por fila, en el mismo orden, incluidos -1 |
| documentos | doc_id posicional, indice_original pandas y tema_id |
| distribucion | Conteos por ID; incluye -1 si existe y suma N |
| atipicos | cantidad, proporcion [0,1], porcentaje [0,100], doc_ids |
| diversidad, coherencia | Diversidad de términos; coherencia no calculada (None) |
| configuracion | Parámetros solicitados/efectivos, revisión E5, versiones y hash del corpus |
| advertencias | Mensaje cuando no hay grupos |

`doc_id` se interpreta con **iloc**, nunca con loc; funciona con índices pandas
no consecutivos o repetidos. `indice_original` es el índice del DataFrame recibido,
no el número de línea del CSV anterior a su depuración. El DataFrame conserva
título, medio, fecha, URL y contenido; todos los IDs son relativos a ese corpus.
Los IDs de temas son los de BERTopic, que puede renumerar los clusters HDBSCAN
por tamaño; no comparar IDs entre ejecuciones como identidades permanentes.
Los representantes son los documentos seleccionados por BERTopic mediante
similitud c-TF-IDF, asociados otra vez a sus posiciones; duplicados textuales
pueden devolver más posiciones que textos representativos.

Mostrar `-1` como «Sin grupo», separado de temas normales. Mostrar None como
«No calculada», no como cero. Si no hay grupos, conservar listado de atípicos y
explicar que la densidad/corpus puede no ofrecer separación útil. No interpretar
pesos c-TF-IDF como probabilidades documentales. No aplicar transform/argmax
de LDA/NMF al objeto BERTopic. `app.py` conserva su interfaz; el responsable
frontend dispone de esta API y del JSON de consola para añadir su vista.

## Contrato para evaluación

Las listas `[t['palabras'] for t in s['temas']]` son compatibles con
`diversidad_tematica`. Se excluye ruido; sin grupos se devuelve None porque no
hay temas evaluables. La diversidad calculada describe estas listas y no prueba
calidad semántica ni comparabilidad con unigramas lematizados.

La NPMI existente divide referencias por espacios: un bigrama no existe como
unidad y las formas flexionadas de BERTopic tampoco coinciden necesariamente con
lemas. Por ello `coherencia=None` y `nota_evaluacion` explica la dependencia.
El frente de evaluación deberá acordar una representación uniforme de descriptores
y corpus de referencia, cobertura de términos, número de palabras y ventana antes
de comparar las tres rutas. No pasar bigramas directamente a la NPMI actual.
Se entregan asignaciones, representantes y parámetros para revisión humana doble
y para barridos; el etiquetado humano y C_v no están implementados aquí.

## Instalación, reproducibilidad y recursos

Desde la raíz, Python 3.11, instalar simultáneamente:
`python -m pip install -r requirements.txt -r requirements-bertopic.txt`.
El archivo opcional fija las versiones principales; sklearn 1.7.2 conserva
`force_all_finite`, utilizado por HDBSCAN 0.8.40. LDA/NMF se verifican también
con esa versión. `pip check` comprueba la resolución instalada.

`python -m src.demo_bertopic` descarga E5 la primera vez en `.cache/huggingface`.
`--offline` resuelve el snapshot a una ruta física y usa exclusivamente esa caché:
esto evita una consulta de metadatos del tokenizador en Transformers 4.57.6
que ignora `local_files_only` al cargar por identificador remoto.
`--csv` acepta el corpus futuro con
las columnas existentes; no requiere cambiar la estructura. El JSON incluye
metadatos originales, resumen y configuración, y se guarda fuera del seguimiento
Git en `output/bertopic/resumen.json`. No contiene un modelo serializado.

Cada resultado registra Python/plataforma, versiones numéricas, parámetros
efectivos, stopwords, hash SHA256 de los textos ordenados y revisión resuelta del
codificador. Por defecto se fija `614241f622f53c4eeff9890bdc4f31cfecc418b3`;
`ConfigBERTopic(revision=...)` permite cambiarla explícitamente.
Semilla fija y UMAP con un hilo mejoran reproducibilidad; diferentes versiones,
CPU/BLAS y operaciones flotantes pueden variar resultados. La primera ejecución
incluye descarga/importación/compilación Numba y puede tardar varios minutos.
La media se calcula por documento y encode trabaja en lotes de 16. No requiere GPU.

## Fuentes técnicas primarias

- [E5: ficha, prefijos, 512 tokens y limitaciones](https://huggingface.co/intfloat/multilingual-e5-small).
- [BERTopic: configuración explícita de HDBSCAN](https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html).
- [BERTopic: vectorizadores y vocabulario](https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html).
- [UMAP: reproducibilidad](https://umap-learn.readthedocs.io/en/latest/reproducibility.html).

## Límites del alcance

El demo contiene 30 noticias sintéticas. Los resultados no son evidencia sobre
la prensa boliviana ni validación sobre el corpus definitivo de 200–400 noticias.
Faltan corpus real, barridos sistemáticos, evaluación homogénea y dos revisores.
Muchos atípicos o pocos grupos pueden ser un resultado legítimo; no se simulan
temas para ocultarlo. No se modificaron los algoritmos ni la evaluación LDA/NMF.
