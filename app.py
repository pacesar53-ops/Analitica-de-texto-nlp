import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import spacy
from wordcloud import WordCloud
from spacy.cli import download
# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ------------------------------------------------------------
st.set_page_config(
    page_title="Analítica Textual - Mensaje de la Nación",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de Analítica Textual usando NLP")
st.caption("Caso de estudio: Análisis de frecuencia, sentimiento y oraciones clave en el Mensaje de la Nación de Keiko Fujimori")

# ------------------------------------------------------------
# 1. CARGA DE MODELO SPACY (Con Caché para mayor velocidad)
# ------------------------------------------------------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load('es_core_news_sm')
    except OSError:
        # Descarga el modelo directamente usando la herramienta oficial de spaCy
        download('es_core_news_sm')
        return spacy.load('es_core_news_sm')

nlp = load_spacy()

# ------------------------------------------------------------
# 2. DICCIONARIOS Y FUNCIONES AUXILIARES
# ------------------------------------------------------------
palabras_positivas = {
    'excelente', 'bueno', 'buena', 'gran', 'optimismo', 'éxito', 'exitoso',
    'mejora', 'favorable', 'crecimiento', 'beneficio', 'ventaja', 'oportunidad',
    'innovación', 'eficiente', 'efectivo', 'logro', 'solución', 'positivo'
}

palabras_negativas = {
    'error', 'fallo', 'problema', 'riesgo', 'pérdida', 'crisis', 'defecto',
    'malo', 'mala', 'amenaza', 'declive', 'retraso', 'peligro', 'negativo',
    'fracaso', 'daño', 'crítico', 'incumplimiento', 'vulnerabilidad'
}

def extraer_lemmas(doc_or_sent, filtrar_longitud=False):
    return [
        token.lemma_.lower() for token in doc_or_sent
        if not (token.is_stop or token.is_punct or token.is_space or token.like_num)
        and (not filtrar_longitud or len(token.lemma_) > 2)
    ]

# ------------------------------------------------------------


# ------------------------------------------------------------
# 4. PROCESAMIENTO Y DASHBOARD
# ------------------------------------------------------------
if texto_raw.strip():
    doc = nlp(texto_raw)
    tokens_limpios = extraer_lemmas(doc, filtrar_longitud=True)

    # Frecuencias
    frecuencia_total = Counter(tokens_limpios)
    freq_positivas = Counter({w: count for w, count in frecuencia_total.items() if w in palabras_positivas})
    freq_negativas = Counter({w: count for w, count in frecuencia_total.items() if w in palabras_negativas})

    df_freq_gen = pd.DataFrame(frecuencia_total.most_common(10), columns=['Palabra', 'Frecuencia'])

    # Clasificación de Oraciones
    analisis_oraciones = []
    frecuencias_oraciones = {}

    for sent in doc.sents:
        texto_oracion = sent.text.strip().replace('\n', ' ')
        if len(texto_oracion) < 5:
            continue

        lemmas_oracion = extraer_lemmas(sent)
        score_pos = sum(1 for lemma in lemmas_oracion if lemma in palabras_positivas)
        score_neg = sum(1 for lemma in lemmas_oracion if lemma in palabras_negativas)

        if score_pos > score_neg:
            polaridad = 'Positiva'
        elif score_neg > score_pos:
            polaridad = 'Negativa'
        else:
            polaridad = 'Neutra'

        analisis_oraciones.append({
            'Oración': texto_oracion,
            'Polaridad': polaridad,
            'Score_Pos': score_pos,
            'Score_Neg': score_neg
        })

        # Puntuación por palabras clave para oraciones relevantes
        tokens_clave = [t for t in sent if not (t.is_stop or t.is_punct or t.is_space or t.like_num)]
        if len(tokens_clave) > 0:
            frecuencias_oraciones[texto_oracion.replace(" ", "_")] = len(tokens_clave)

    df_oraciones = pd.DataFrame(analisis_oraciones)

    # ------------------------------------------------------------
    # MÉTRICAS CLAVE (KPIs)
    # ------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Palabras Analizadas", len(tokens_limpios))
    col2.metric("Palabras Positivas", sum(freq_positivas.values()))
    col3.metric("Palabras Negativas", sum(freq_negativas.values()))
    col4.metric("Total Oraciones", len(df_oraciones))

    st.markdown("---")

    # ------------------------------------------------------------
    # SECCIÓN 1: VISUALIZACIONES PRINCIPALES (GRID 2x2)
    # ------------------------------------------------------------
    st.subheader("📈 Frecuencia de Palabras")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    # 1. WordCloud
    texto_para_nube = " ".join(tokens_limpios)
    if texto_para_nube:
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            colormap='Blues', max_words=100, collocations=False
        ).generate(texto_para_nube)
        axes[0, 0].imshow(wordcloud, interpolation='bilinear')
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Nube de Palabras Frecuentes', fontweight='bold', fontsize=12)

    # 2. Top 10 Palabras
    sns.barplot(
        data=df_freq_gen, x='Frecuencia', y='Palabra',
        ax=axes[0, 1], hue='Palabra', palette='Blues_r', legend=False
    )
    axes[0, 1].set_title('Top 10 Palabras Más Frecuentes', fontweight='bold', fontsize=12)

    # 3. Positivas vs Negativas
    df_palabras_sentimiento = pd.DataFrame({
        'Tipo': ['Positivas', 'Negativas'],
        'Total': [sum(freq_positivas.values()), sum(freq_negativas.values())]
    })
    sns.barplot(
        data=df_palabras_sentimiento, x='Tipo', y='Total',
        ax=axes[1, 0], hue='Tipo', palette=['#2ecc71', '#e74c3c'], legend=False
    )
    axes[1, 0].set_title('Total Palabras Positivas vs Negativas', fontweight='bold', fontsize=12)

    # 4. Clasificación de Oraciones
    sns.countplot(
        data=df_oraciones, x='Polaridad',
        ax=axes[1, 1], hue='Polaridad',
        palette={'Positiva': '#2ecc71', 'Negativa': '#e74c3c', 'Neutra': '#95a5a6'},
        legend=False
    )
    axes[1, 1].set_title('Clasificación de Oraciones por Polaridad', fontweight='bold', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ------------------------------------------------------------
# SECCIÓN 2: ACORDEONES DESPLEGABLES
    # ------------------------------------------------------------
    st.subheader("📌 Top Oraciones con Mayor Relevancia Informativa")

    df_oraciones_score = pd.DataFrame([
        {'Oración': key.replace('_', ' '), 'Peso / Score': val}
        for key, val in frecuencias_oraciones.items()
    ]).sort_values(by='Peso / Score', ascending=False).head(10).reset_index(drop=True)

    for idx, row in df_oraciones_score.iterrows():
        # Título resumido para el expander
        resumen = row['Oración'][:90] + "..." if len(row['Oración']) > 90 else row['Oración']
        
        with st.expander(f"**#{idx + 1}** (Score: {row['Peso / Score']}) — {resumen}"):
            st.write(row['Oración'])

else:
    st.info("👆 Por favor, carga un archivo `.txt` o escribe algún texto en la barra lateral para iniciar el análisis.")
