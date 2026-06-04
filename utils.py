import streamlit as st
import pandas as pd
import numpy as np
import pickle
import spacy
import re

# Custom CSS for modern/premium aesthetics
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Overall Font Styling */
html, body, [class*="css"], .stText, .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
}

/* Normal Text Font Size */
p, li, .stMarkdown p, .stMarkdown li {
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
    color: #334155 !important;
}

/* Main Background */
.stApp {
    background-color: #f8fafc;
}

/* Header Styling */
.header-banner {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
}
.header-banner h1 {
    margin: 0;
    font-weight: 700;
    font-size: 2.2rem;
    color: white !important;
}
.header-banner p {
    margin: 10px 0 0 0;
    font-size: 1.1rem;
    color: white !important;
    opacity: 0.9;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
    padding-top: 20px;
}

/* Metric Card Styling */
.custom-card {
    background: white;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 20px;
}
.custom-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
}
.metric-label {
    font-size: 0.85rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
}
.metric-sub {
    font-size: 0.85rem;
    color: #10b981;
    margin-top: 4px;
    font-weight: 500;
}

/* Prediction Output Card Styling */
.prediction-card {
    padding: 24px;
    border-radius: 12px;
    color: white;
    margin-top: 15px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.sports-theme {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}
.business-theme {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}
.scitech-theme {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
}
.world-theme {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
}

/* List Highlights */
ul.highlights-list {
    list-style: none;
    padding-left: 0;
}
ul.highlights-list li {
    padding-left: 28px;
    position: relative;
    margin-bottom: 12px;
    font-size: 1.05rem;
    color: #334155;
}
ul.highlights-list li::before {
    content: "✓";
    position: absolute;
    left: 0;
    color: #10b981;
    font-weight: bold;
    font-size: 1.2rem;
}

/* custom buttons */
.stButton>button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
}
.stButton>button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 12px -1px rgba(79, 70, 229, 0.3) !important;
}
.stButton>button[data-testid="stBaseButton-secondary"] {
    background-color: #ffffff !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.stButton>button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-2px) !important;
}
</style>
"""

# Helper function to render a premium metric card
def render_metric_card(label, value, subtext=None):
    sub_html = f"<div class='metric-sub'>{subtext}</div>" if subtext else ""
    st.markdown(f"""
        <div class="custom-card">
            <div class="metric-label">{label}</div>
            <div class="metric-val">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_dataset():
    # Only load required columns to optimize speed and memory usage
    return pd.read_csv("agnews_clean.csv", usecols=['topic', 'word_count', 'char_count', 'clean_text'])

# Cache model and vectorizer loading
@st.cache_resource
def load_models():
    with open("svm.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)
    return svm_model, tfidf_vec

# Load spaCy NLP model
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to dynamic model installation if not pre-installed (e.g. local vs cloud build environments)
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")

# Preprocessing pipeline
def preprocess_text(text, nlp):
    # Remove standard publisher headers (Reuters, AFP, AP) at the start
    text = re.sub(r'^(reuters|afp|ap)\s*-\s*', '', text, flags=re.IGNORECASE)
    # Replace backslashes and hyphens with space
    text = re.sub(r'[\\-]', ' ', text)
    # Process text using spaCy
    doc = nlp(text)
    # Filter stopwords, punctuation and whitespace, then lemmatize
    tokens = []
    for token in doc:
        if not token.is_punct and not token.is_stop and not token.is_space:
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)

# Topic mapping constants
topic_prettymap = {
    'world': 'World',
    'sports': 'Sports',
    'business': 'Business',
    'sci_tech': 'Science/Technology'
}

topic_colors = {
    'world': '#06b6d4',
    'sports': '#10b981',
    'business': '#f59e0b',
    'sci_tech': '#6366f1'
}

import plotly.io as pio

# Configure default plotly layout for premium styling with larger fonts
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font.family = "Outfit"
pio.templates["plotly_white"].layout.font.size = 14
pio.templates["plotly_white"].layout.title.font.size = 18

def render_plotly_chart(fig, use_container_width=True):
    # Explicitly enforce custom fonts/sizes
    fig.update_layout(
        font=dict(family="Outfit", size=14),
        title_font=dict(family="Outfit", size=18)
    )
    # Pass theme=None to prevent Streamlit from overriding our custom font configurations
    st.plotly_chart(fig, use_container_width=use_container_width, theme=None)

