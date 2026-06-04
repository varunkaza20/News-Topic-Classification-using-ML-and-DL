import streamlit as st
from utils import render_metric_card

def set_page(page_name):
    st.session_state.current_page = page_name

def render_home_page():
    st.markdown("""
        <div class="header-banner">
            <h1>News Topic Classification Using Machine Learning and Deep Learning</h1>
            <p>Comparing traditional machine learning and deep learning approaches to categorize news articles into predefined topics</p>
        </div>
    """, unsafe_allow_html=True)

    # Quick Navigation Buttons below the title banner
    st.markdown("<h4 style='color: #1e293b; font-weight: 600; margin-bottom: 12px;'>Quick Navigation</h4>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        st.button(
            "Exploratory Data Analysis", 
            use_container_width=True, 
            key="home_nav_eda",
            on_click=set_page,
            args=("Exploratory Data Analysis",)
        )
    with nav_col2:
        st.button(
            "Topic Prediction", 
            use_container_width=True, 
            key="home_nav_pred",
            on_click=set_page,
            args=("Topic Prediction",)
        )
    with nav_col3:
        st.button(
            "Model Comparison", 
            use_container_width=True, 
            key="home_nav_compare",
            on_click=set_page,
            args=("Model Comparison",)
        )
    with nav_col4:
        st.button(
            "About Project", 
            use_container_width=True, 
            key="home_nav_about",
            on_click=set_page,
            args=("About Project",)
        )
            
    st.markdown("<hr style='border-top: 1px solid #cbd5e1; margin: 25px 0;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<h3 style='color: #1e293b; font-weight: 600;'>Project Overview</h3>", unsafe_allow_html=True)
        st.write(
            "News articles are generated at an enormous scale every day across digital platforms. "
            "Automatically categorizing news into predefined topics is an important Natural Language Processing (NLP) task "
            "that helps news aggregators, search engines, recommendation systems, and media organizations organize content efficiently.\n\n"
            "This project develops an end-to-end News Topic Classification system capable of automatically assigning a news "
            "article to one of four categories: World, Sports, Business, and Sci/Tech. The project explores both traditional "
            "machine learning and deep learning approaches to understand how different techniques perform on short news articles. "
            "The complete workflow includes data preprocessing, exploratory data analysis, feature engineering, model training, "
            "performance evaluation, error analysis, and deployment through this interactive Streamlit application."
        )
        
        st.markdown("<h3 style='color: #1e293b; font-weight: 600; margin-top: 20px;'>Key Highlights</h3>", unsafe_allow_html=True)
        st.markdown("""
            <ul class="highlights-list">
                <li><b>Data Preprocessing:</b> Cleaned text using lowercase conversion, URL removal, special character removal, publisher news source artifacts filtering (e.g. Reuters, AFP), tokenization, lemmatization using spaCy, and stopword removal using NLTK.</li>
                <li><b>Exploratory Data Analysis:</b> In-depth analysis of class balance, word/character count distributions, and topic vocabularies.</li>
                <li><b>Feature Engineering:</b> TF-IDF configurations utilizing unigrams + bigrams and a vocabulary limit of 20,000 features.</li>
                <li><b>Model Architectures:</b> Evaluated Multinomial Naive Bayes, Logistic Regression, Linear Support Vector Machine (SVM), and Bidirectional LSTM (BiLSTM).</li>
                <li><b>Real-Time Inference:</b> Interactive prediction system utilizing the best performing Linear SVM model for unseen text.</li>
                <li><b>Deployment:</b> Interactive dashboards showcasing model comparisons, error analysis, and EDA.</li>
            </ul>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("<h3 style='color: #1e293b; font-weight: 600; text-align: center;'>Key Metrics</h3>", unsafe_allow_html=True)
        
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            render_metric_card("Dataset Size", "120,000", "Articles in clean set")
            render_metric_card("Best Accuracy", "90.49%", "Achieved by Linear SVM")
        with col_sub2:
            render_metric_card("Number of Classes", "4 Topics", "World, Sports, Business, Tech")
            render_metric_card("Vocabulary Size", "46,737", "Total unique words")
