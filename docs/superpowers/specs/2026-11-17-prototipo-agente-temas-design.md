# Diseño: Prototipo de Agente de Descubrimiento de Temas — Grupo 15

Fecha: 2026-11-17 · Estado: **aprobado por el usuario**

## Intención y porqué

Prototipo académico funcional (primer avance, gestión II-2026) del agente de análisis documental del Grupo 15: descubrir temas recurrentes en noticias bolivianas en español mediante aprendizaje no supervisado, para demostrar ante el docente cómo funcionan LDA y NMF con resultados reales. Síntesis del informe `Primer_avance_IA_Grupo_15_BLANCO_Y_NEGRO.docx` (secciones 4.1–4.5) y del enunciado del spec del usuario.

## Objetivos

- App Streamlit "Explorador Inteligente de Temas – Grupo 15": cargar noticias (CSV/TXT), preprocesar, entrenar LDA (conteos) y NMF (TF-IDF), mostrar temas, palabras, noticias representativas, comparar y evaluar.
- Pruebas pytest que prueben carga, preproceso, matrices, modelos, temas y manejo de errores.
- Corpus sintético claramente identificado (~30 noticias, 6 temas, medios ficticios "Medio Demo 1/2").

## No-objetivos y anti-objetivos

- No BERTopic (ampliación futura del informe), ni despliegue, ni usuarios, ni base de datos, ni claves API.
- Anti-objetivo: cualquier resultado simulado o pantalla decorativa; cada número mostrado debe salir de un modelo entrenado en la sesión.
- Anti-objetivo: atribuir textos sintéticos a medios reales.

## Restricciones

- Tecnologías obligatorias: Python, Streamlit, scikit-learn, pandas, numpy, CountVectorizer (LDA), TF-IDF (NMF), librería PLN español.
- Semilla fija 42; LDA batch y NMF nndsvda para reproducibilidad.
- Informe 3.6: no eliminar tildes («público»/«publicó»); lematización cuidada; stopwords editoriales revisables.
- Informe 3.13: comparación de modelos con la misma medida de coherencia (NPMI propia sobre el corpus, sin gensim) y diversidad = proporción de términos únicos.
- Ejecutable en Windows + VS Code (instrucciones); verificación local en macOS (Python 3.15 con wheels OK).

## Enfoques considerados

1. **spaCy `es_core_news_sm` para lematización** (elegido): lemas reales en español, gratuito, instaló sin problemas en Python 3.15. Descartado NLTK (sin lematizador bueno para español) y stemming puro (fragmenta términos: «diésel»/«diésels»).
2. **Coherencia NPMI implementada en casa con NumPy** (elegido): medida común a LDA y NMF sobre el propio corpus, sin dependencia de gensim (instalación frágil en Windows/Python nuevo). Descartada gensim CoherenceModel (Cv) por ese riesgo; el informe pide "una medida común", no Cv específicamente.
3. **Módulos planos + `pipeline.py` como orquestador** (elegido): la app y los tests de integración consumen un único `ejecutar_pipeline(df, n_temas)`; sin clases innecesarias.

## Decisiones del "cómo"

- Corpus demo generado por script determinista (`datos/generar_noticias_demo.py`, semilla 2026) + `LEEME.txt` que declara su carácter sintético.
- Errores de dominio con mensajes accionables: `DatosInvalidosError`, `CorpusInvalidoError`, `CorpusInsuficienteError`.
- UI con sidebar de 6 páginas y estado en `st.session_state`; helpers puros de presentación testeables (`tabla_temas`, `etiqueta_tema`).
- Defaults: `n_temas=6` (slider 2–10), `n_palabras=10`, vocabulario `min_df=2, max_df=0.9, max_features=3000` idéntico para ambas rutas (comparabilidad del informe 4.4).

## Lo que queda al implementador

Cuerpos de funciones con firma y tests fijados en el plan; bancos de frases exactos del corpus demo; copy fino de la UI.
