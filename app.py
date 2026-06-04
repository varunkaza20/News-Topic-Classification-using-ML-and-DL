import streamlit as st
from utils import (
    CUSTOM_CSS,
    load_dataset,
    load_models,
    load_spacy
)
from pages.home import render_home_page
from pages.eda import render_eda_page
from pages.prediction import render_prediction_page
from pages.comparison import render_comparison_page
from pages.about import render_about_page

# Set page config
st.set_page_config(
    page_title="News Topic Classification",
    page_icon="📰",
    layout="wide"
)

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Load data and models (cached)
df = load_dataset()
svm, tfidf = load_models()
nlp = load_spacy()

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

pages = ["Home", "Exploratory Data Analysis", "Topic Prediction", "Model Comparison", "About Project"]

# Callback to set page
def set_page(page_name):
    st.session_state.current_page = page_name

# Render buttons in sidebar
for p in pages:
    is_active = st.session_state.current_page == p
    st.sidebar.button(
        p, 
        use_container_width=True, 
        key=f"nav_{p}", 
        type="primary" if is_active else "secondary",
        on_click=set_page,
        args=(p,)
    )

# Dispatch rendering based on active page
page = st.session_state.current_page

if page == "Home":
    render_home_page()
elif page == "Exploratory Data Analysis":
    render_eda_page(df)
elif page == "Topic Prediction":
    render_prediction_page(svm, tfidf, nlp)
elif page == "Model Comparison":
    render_comparison_page(svm, tfidf)
elif page == "About Project":
    render_about_page()

# Footer
st.markdown("<hr style='border-top: 1px solid #cbd5e1; margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.95rem; font-weight: 500;'>Made with ❤️ by Varun Kaza</p>", unsafe_allow_html=True)
