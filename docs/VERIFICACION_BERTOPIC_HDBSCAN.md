# Evidencia de verificación BERTopic/HDBSCAN

Ejecutado el 24/11/2026 en Windows, Python 3.11.9, rama
`feature/bertopic-hdbscan`, base `0cfc1bf`.

## Comandos y resultados observados

| Ejecución | Resultado |
|---|---|
| `python -m pytest -q` antes de cambios | 59 passed, 27.08 s |
| Primera suite ampliada sin integración activada | 80 passed, 1 skipped, 39.27 s |
| Primer intento de integración offline | 82 passed, 1 failed: Transformers consultaba Hub pese a local_files_only |
| Suite completa tras corregir carga offline, con RUN_BERTOPIC_INTEGRATION=1 | 83 passed, 54.90 s; 2 FutureWarning de HDBSCAN/sklearn |
| Integración demo ampliada después con fragmentación y tokenizador E5 real | 1 passed, 43.13 s |
| `python -m src.demo_bertopic --offline` | Entrenamiento real y exportación JSON correctos |
| `python -m pip check` | No broken requirements found |
| `git diff --check` | Sin errores |

Los avisos de HDBSCAN corresponden al nombre obsoleto `force_all_finite`:
no son fallos; se fija sklearn 1.7.2 compatible. La suite ordinaria omite
explícitamente dos pruebas de integración si no se activa la variable.

La revisión del primer JSON detectó pérdida de letras acentuadas por el modo
inglés predeterminado de BERTopic. Se corrigió con `language="multilingual"`
y una prueba real exige diésel/sequía/dólar/fútbol en el vocabulario.
El problema offline se corrigió resolviendo la ruta física del snapshot antes
de cargar SentenceTransformer. La suite final se ejecutó sin permiso de red.

## Demo observado (no datos de prensa real)

Entrada: `datos/noticias_demo.csv`, 30 noticias sintéticas ya existentes,
sin modificar el CSV ni su generador. Salida completa local:
`output/bertopic/resumen.json`, excluida de Git (contiene documentos y parámetros).

- 30 documentos analizados.
- 5 temas descubiertos (excluyendo -1).
- 0 atípicos, 0 %; cobertura 100 %.
- Distribución por ID: `{0: 10, 1: 5, 2: 5, 3: 5, 4: 5}`.
- Diversidad de los 10 descriptores por tema: 1.0.
- Coherencia NPMI: no calculada; pendiente de representación comparable.

| ID | Noticias | Primeros cinco descriptores reales |
|---|---:|---|
| 0 | 10 | salud, vacunación, educación, casos, ministerio |
| 1 | 5 | combustible, abastecimiento, gasolina, transporte, abastecimiento combustible |
| 2 | 5 | agua, sequía, lluvias, falta, miles familias |
| 3 | 5 | dólar, precio, precios, mercados, advierten presión |
| 4 | 5 | partido, fútbol, torneo, goles, partido goles |

El grupo 0 reúne las noticias sintéticas de salud y educación (posiciones 20–29).
No se recuperan como seis grupos separados los seis asuntos del generador.
La diversidad 1.0 significa términos distintos entre listas, no separación
semántica perfecta. No se cambiaron parámetros para forzar seis temas.

Los artículos demo caben en un fragmento cada uno. La fragmentación se verifica
también con un texto largo controlado; el corpus definitivo todavía no se evaluó.

E5: revisión `614241f622f53c4eeff9890bdc4f31cfecc418b3`, CPU, prefijo query,
512 tokens máximos por fragmento. Hash SHA256 de los textos normalizados y ordenados:
`3e7ab3882e93b9a50cb47460f019565425eade0ba0f1130e5c93f3a71094cb91`.
Versiones principales fijadas en `requirements-bertopic.txt`; spaCy 3.8.18.

## Qué verifican las pruebas

- La integración demo ejecuta el codificador E5 real y comprueba instancias
  y atributos ajustados de UMAP, HDBSCAN y ClassTfidfTransformer.
- Máscara de ruido HDBSCAN idéntica a asignaciones -1 de BERTopic, 30 resultados
  y grupos, vocabulario y representantes derivados del modelo.
- Corpus mínimo de cinco textos: embeddings controlados de prueba, pero
  UMAP/HDBSCAN/c-TF-IDF reales; produce cero grupos y cinco atípicos (100 %).
  Este caso no se presenta como experimento semántico con E5.
- Pruebas unitarias con dobles para contratos, todos/ningún/algunos atípicos,
  índices repetidos, posiciones, entradas inválidas y fragmentación sin pérdida.
- Selección de LDA/NMF no llama BERTopic; selección de BERTopic no ejecuta
  preprocesamiento léxico. Selección combinada conserva exactamente resúmenes
  y componentes LDA/NMF respecto a la llamada original (BERTopic simulado
  únicamente en esta prueba de aislamiento).

No se hicieron pruebas de interfaz visual nueva (no se modificó `app.py`),
ni evaluación sobre 200–400 noticias reales, ni revisión humana doble.
La responsabilidad de esta rama entrega API y contrato para esos frentes.

## Inventario de cambios y Git

Nuevos: `src/bertopic_modelo.py`, `src/demo_bertopic.py`,
`tests/test_bertopic.py`, `requirements-bertopic.txt`,
`docs/DECISION_TECNICA_BERTOPIC_HDBSCAN.md` y este registro.

Modificados: `src/pipeline.py` (selección opcional), `.gitignore` (caché/salida),
`requirements.txt` (referencia a instalación opcional), `README.md`
(instalación/ejecución) y `docs/explicacion_academica.md` y
`docs/preparacion_defensa.md` (actualización de alcance).

Estado de entrega: seis archivos modificados y seis nuevos sin commit,
en `feature/bertopic-hdbscan`. Sin push, fusión ni Pull Request.
`app.py`, algoritmos LDA/NMF, evaluación y datos demo no fueron modificados.
Git ignora `.venv`, `.cache` y `output/bertopic`.
