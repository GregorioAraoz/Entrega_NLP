import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from preprocess import clean_text
from features import load_vectorizer, transform_text
from models_deep import load_model as load_deep_model
from insights import calculate_global_score, get_top_ngrams, extract_opportunities

# Page Config
st.set_page_config(page_title="Sentimentalyst - Restaurantes", layout="wide", page_icon="🍔")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #eee;
    }
    
    .aspect-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        color: white;
        margin-right: 5px;
    }
    .badge-pos { background-color: #00C851; }
    .badge-neu { background-color: #ffbb33; }
    .badge-neg { background-color: #ff4444; }

    h1 {
        background: -webkit-linear-gradient(45deg, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Cargando Sentimentalyst v3.0 (Deep Learning)...")
def load_resources():
    resources = {}
    resources['vec_bow'] = load_vectorizer('models/vectorizer_bow.pkl')
    resources['vec_tfidf'] = load_vectorizer('models/vectorizer_tfidf.pkl')
    # Load Label Encoder (Simulated with fixed map to match training)
    # 0: NEG, 1: NEU, 2: POS
    resources['le_map'] = {0: 'NEG', 1: 'NEU', 2: 'POS'}

    # Load Deep Model Only
    # Determine input dim from vectorizer
    if resources['vec_tfidf']:
        input_dim = len(resources['vec_tfidf'].get_feature_names_out())
        resources['model_deep'] = load_deep_model(input_dim, 'models/deep_mlp.pth')
    else:
        resources['model_deep'] = None
        
    return resources

resources = load_resources()

with st.sidebar:
    st.title("🍔 Restaurant Analytics")
    st.markdown("Analizador de Sentimiento Multi-Aspecto")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Subir reseñas (CSV)", type=['csv'])
    
    if uploaded_file is None:
        st.info("👆 Por favor sube un archivo CSV con columnas 'place_name' y 'review_text' para comenzar.")
        st.stop()
    else:
        df = pd.read_csv(uploaded_file)
    
    if 'place_name' not in df.columns or 'review_text' not in df.columns:
        st.error("CSV debe tener: 'place_name', 'review_text'")
        st.stop()
        
    places = df['place_name'].unique()
    selected_place = st.selectbox("Lugar", places)
    place_df = df[df['place_name'] == selected_place].copy()
    
    st.markdown("---")
    # Model selection hidden - defaulted to Deep Learning for best performance
    model_choice = "Deep Learning (MLP)"
    
    show_raw_data = st.checkbox("🔍 Ver datos crudos", value=False)

# Main Inference Logic
if 'sentiment_global' not in place_df.columns:
    with st.spinner('🔮 Analizando sentimientos...'):
        place_df['cleaned_text'] = place_df['review_text'].apply(clean_text)
        place_df = place_df[place_df['cleaned_text'] != ""]
        
        if resources['model_deep'] and resources['vec_tfidf']:
            X = transform_text(place_df['cleaned_text'], resources['vec_tfidf'])
            X_t = torch.tensor(X.toarray(), dtype=torch.float32)
            probs_dict = resources['model_deep'].predict_proba(X_t)
            
            for aspect, probs in probs_dict.items():
                _, idxs = torch.max(probs, 1)
                # Use fixed map
                labels = [resources['le_map'].get(i.item(), 'NEU') for i in idxs]
                place_df[f'sentiment_{aspect}'] = labels
        else:
            st.error("Grave: El modelo Deep Learning no se pudo cargar. Revisa que los archivos en /models existan.")
            st.stop()
else:
    # already labeled
    if 'cleaned_text' not in place_df.columns:
        place_df['cleaned_text'] = place_df['review_text'].apply(clean_text)

# --- UI ---
# --- Logic: Override Global Sentiment based on User Rule ---
# "Si menciona algo positivo pero otra cosa negativa -> NEU"
def apply_global_rule(row):
    pos = 0
    neg = 0
    for k in ['food', 'service', 'price']:
        if f'sentiment_{k}' in row:
            if row[f'sentiment_{k}'] == 'POS': pos += 1
            elif row[f'sentiment_{k}'] == 'NEG': neg += 1
    
    if pos > 0 and neg > 0: return 'NEU'
    if pos > 0: return 'POS'
    if neg > 0: return 'NEG'
    return 'NEU'

if 'sentiment_food' in place_df.columns:
    place_df['sentiment_global'] = place_df.apply(apply_global_rule, axis=1)

# --- UI ---
st.title(f"📍 {selected_place}")

# Helper for colors
def get_color(score):
    return "#00C851" if score >= 60 else "#FFbb33" if score >= 40 else "#ff4444"

# KPIs
score, rating = calculate_global_score(place_df, 'sentiment_global')
total = len(place_df)
pos_count = len(place_df[place_df['sentiment_global'] == 'POS'])
neu_count = len(place_df[place_df['sentiment_global'] == 'NEU'])
neg_count = len(place_df[place_df['sentiment_global'] == 'NEG'])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Global Score", f"{score}/100", rating)
k2.metric("Total Reseñas", total)
k3.metric("Positivas", pos_count, f"{pos_count/total:.1%}" if total else "")
k4.metric("Neutras", neu_count, f"{neu_count/total:.1%}" if total else "", delta_color="off")
k5.metric("Negativas", neg_count, f"{neg_count/total:.1%}" if total else "", delta_color="inverse")

st.markdown("---")
cols = st.columns(3)
aspects = [('food', '🍕 Comida'), ('service', '💁 Atención'), ('price', '💰 Precio')]

for i, (key, label) in enumerate(aspects):
    # Calculate score based ONLY on relevant opinions (ignoring neutrals)
    # This addresses the user's concern about "dilution" by unmentioned aspects
    col_data = place_df[f'sentiment_{key}']
    relevant = col_data[col_data != 'NEU']
    count = len(relevant)
    
    if count > 0:
        pos_count = (relevant == 'POS').sum()
        score = (pos_count / count) * 100
    else:
        score = 50 # Default neutral if no specific opinions
    
    with cols[i]:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 5px solid {get_color(score)}">
            <h3>{label}</h3>
            <h2 style="color:{get_color(score)}">{score:.0f}/100</h2>
            <p style="margin-bottom:0;">{score:.0f}% Aprobación</p>
            <p style="font-size: 0.8em; color: gray; margin-top:5px;">({count} opiniones)</p>
        </div>
        """, unsafe_allow_html=True)

# Main Insights (Focus on Global for keywords, but maybe filtered?)
st.markdown("---")

filter_aspect = st.selectbox("🔍 Profundizar en:", 
                             ['Global', 'Comida (Food)', 'Atención (Service)', 'Precio (Price)'], 
                             index=0)

aspect_map = {
    'Global': 'global',
    'Comida (Food)': 'food',
    'Atención (Service)': 'service',
    'Precio (Price)': 'price'
}
target_col = f"sentiment_{aspect_map[filter_aspect]}"

col_kw, col_rev = st.columns([1, 1])

with col_kw:
    st.subheader(f"🗣️ Lo que dicen de: {filter_aspect}")
    
    # Filter reviews where this specific aspect is POS or NEG
    pos_reviews = place_df[place_df[target_col] == 'POS']['cleaned_text']
    neg_reviews = place_df[place_df[target_col] == 'NEG']['cleaned_text']
    
    top_pos = get_top_ngrams(pos_reviews, n=2, top_k=8)
    top_neg = get_top_ngrams(neg_reviews, n=2, top_k=8)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 👍 Lo Bueno")
        if top_pos:
            for txt, freq in top_pos:
                st.success(f"{txt} ({freq})")
        else: st.info("Sin datos.")
        
    with c2:
        st.markdown("#### 👎 Lo Malo")
        if top_neg:
            for txt, freq in top_neg:
                st.error(f"{txt} ({freq})")
        else: st.info("Sin datos.")

with col_rev:
    st.subheader(f"📝 Reseñas Relevantes ({filter_aspect})")
    # Show badge for Aspects
    # Filter snippets that actually talk about this aspect (heuristic: label is not NEU)
    relevant_subset = place_df[place_df[target_col] != 'NEU']
    if relevant_subset.empty: relevant_subset = place_df # Fallback
    
    # Show 5 random samples initially
    sample = relevant_subset.sample(min(5, len(relevant_subset)))
    for _, row in sample.iterrows():
        # Build badges
        badges = ""
        for k in ['food', 'service', 'price']:
            if f'sentiment_{k}' in row:
                sent = row[f'sentiment_{k}']
                if sent == 'POS': badges += f"<span class='aspect-badge badge-pos'>{k.capitalize()}</span>"
                elif sent == 'NEG': badges += f"<span class='aspect-badge badge-neg'>{k.capitalize()}</span>"
        
        st.markdown(f"""
        <div style="background:white; padding:10px; border-radius:10px; margin-bottom:10px; border-left:3px solid #ccc;">
            <div style="margin-bottom:5px;">{badges}</div>
            <i>"{row['review_text']}"</i>
        </div>
        """, unsafe_allow_html=True)

    # Allow viewing ALL
    with st.expander(f"👁️ Ver todas ({len(relevant_subset)})", expanded=False):
        for _, row in relevant_subset.iterrows():
            badges = ""
            for k in ['food', 'service', 'price']:
                if f'sentiment_{k}' in row:
                    sent = row[f'sentiment_{k}']
                    if sent == 'POS': badges += f"<span class='aspect-badge badge-pos'>{k.capitalize()}</span>"
                    elif sent == 'NEG': badges += f"<span class='aspect-badge badge-neg'>{k.capitalize()}</span>"
            
            st.markdown(f"""
            <div style="background:white; padding:10px; border-radius:10px; margin-bottom:10px; border-left:3px solid #eee;">
                <div style="margin-bottom:5px;">{badges}</div>
                <p style="margin:0;">"{row['review_text']}"</p>
            </div>
            """, unsafe_allow_html=True)

if show_raw_data:
    st.dataframe(place_df)
