import streamlit as st
import pandas as pd
import plotly.express as px
from utils import topic_colors, topic_prettymap, render_plotly_chart

def render_comparison_page(svm, tfidf):
    st.markdown("""
        <div class="header-banner">
            <h1>Model Performance Comparison</h1>
            <p>Interactive accuracy evaluation of classical machine learning and BiLSTM models</p>
        </div>
    """, unsafe_allow_html=True)
    
    acc_data = pd.DataFrame({
        'Model': ['Naive Bayes', 'Logistic Regression', 'Linear SVM', 'BiLSTM'],
        'Accuracy (%)': [89.74, 90.35, 90.49, 90.00]
    })
    
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown("<h3 style='color: #1e293b; font-weight: 600;'>Accuracy Comparison</h3>", unsafe_allow_html=True)
        fig = px.bar(
            acc_data, x='Model', y='Accuracy (%)',
            color='Model',
            color_discrete_map={
                'Naive Bayes': '#94a3b8',
                'Logistic Regression': '#6366f1',
                'Linear SVM': '#ef4444', # Highlight the best model in red/coral
                'BiLSTM': '#06b6d4'
            },
            range_y=[85, 92]
        )
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        render_plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("<h3 style='color: #1e293b; font-weight: 600;'>Performance Table</h3>", unsafe_allow_html=True)
        st.dataframe(acc_data, hide_index=True, use_container_width=True)
        
        # Best model highlight card
        st.markdown("""
            <div class="custom-card" style="border-left: 5px solid #ef4444; background-color: #fef2f2;">
                <div class="metric-label" style="color: #ef4444;">Top Classifier</div>
                <div class="metric-val" style="font-size: 1.4rem;">Linear SVM</div>
                <div class="metric-sub" style="color: #ef4444; font-size: 0.9rem;">
                    Highest classification accuracy of <b>90.49%</b> using TF-IDF bigrams. 
                    Interestingly, the deep learning model (BiLSTM) did not significantly outperform 
                    traditional machine learning approaches despite its higher computational complexity.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h3 style='color: #1e293b; font-weight: 600; margin-top: 25px;'>Key Findings</h3>", unsafe_allow_html=True)
    
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
            <div class="custom-card">
                <h5 style='color: #6366f1; margin:0 0 8px 0; font-weight: 600;'>Finding 1: Traditional ML Competitiveness & SVM Efficiency</h5>
                <p style='color: #334155; font-size: 0.95rem; margin:0;'>
                    Traditional machine learning models remain highly competitive for short-text classification. 
                    Linear SVM achieved the best overall performance (90.49%) while requiring significantly less 
                    computational resources than BiLSTM.
                </p>
            </div>
            <div class="custom-card">
                <h5 style='color: #f59e0b; margin:0 0 8px 0; font-weight: 600;'>Finding 3: Semantic Overlap</h5>
                <p style='color: #334155; font-size: 0.95rem; margin:0;'>
                    Business and Sci/Tech categories exhibit substantial semantic overlap, making them the most challenging classes to separate.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with f2:
        st.markdown("""
            <div class="custom-card">
                <h5 style='color: #ef4444; margin:0 0 8px 0; font-weight: 600;'>Finding 2: BiLSTM Deep Learning Overfitting</h5>
                <p style='color: #334155; font-size: 0.95rem; margin:0;'>
                    The BiLSTM model showed clear signs of overfitting. Validation accuracy plateaued quickly due to 
                    the short length of descriptions, indicating that complex recurrent nets may overfit on short-text 
                    news samples without massive datasets.
                </p>
            </div>
            <div class="custom-card">
                <h5 style='color: #06b6d4; margin:0 0 8px 0; font-weight: 600;'>Finding 4: Keyword-Based Effectiveness</h5>
                <p style='color: #334155; font-size: 0.95rem; margin:0;'>
                    For short news descriptions, keyword-based representations such as TF-IDF remain extremely effective.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Error Analysis Section
    st.markdown("<h3 style='color: #1e293b; font-weight: 600; margin-top: 25px;'>Error Analysis</h3>", unsafe_allow_html=True)
    st.write("Incorrectly classified samples were manually analyzed to identify major sources of classification errors:")
    
    err1, err2, err3 = st.columns(3)
    with err1:
        st.markdown("""
            <div class="custom-card" style="height: 100%;">
                <h5 style='color: #ef4444; margin:0 0 8px 0; font-weight: 600;'>Business ↔ Sci/Tech Confusion</h5>
                <p style='color: #334155; font-size: 0.95rem; margin:0;'>
                    The largest source of classification errors occurred between Business and Sci/Tech. 
                    Articles involving technology company earnings, product launches, telecommunications announcements, 
                    and corporate acquisitions naturally contain overlapping vocabulary belonging to both categories.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with err2:
        st.markdown("""
            <div class="custom-card" style="height: 100%;">
                <h5 style='color: #f59e0b; margin:0 0 8px 0; font-weight: 600;'>World ↔ Business Confusion</h5>
                <p style='color: #334155; font-size: 0.95rem; margin:0;'>
                    Articles involving international trade, economic agreements, energy markets, and geopolitical developments 
                    often overlapped and caused misclassifications between World and Business topics.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with err3:
        st.markdown("""
            <div class="custom-card" style="height: 100%;">
                <h5 style='color: #10b981; margin:0 0 8px 0; font-weight: 600;'>Sports High Separability</h5>
                <p style='color: #334155; font-size: 0.95rem; margin:0;'>
                    Sports articles demonstrated the highest separability and lowest error rates because of 
                    highly distinctive category vocabulary and limited overlap with other fields.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # SVM Feature Importances
    st.markdown("<hr style='border-top: 1px solid #cbd5e1; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1e293b; font-weight: 600;'>SVM Feature Importances</h3>", unsafe_allow_html=True)
    st.write("Below are the words with the highest positive coefficients in the Linear SVM model for each of the four categories:")

    pretty_to_raw = {
        'Business': 'business',
        'Sci/Tech': 'sci_tech',
        'Sports': 'sports',
        'World': 'world'
    }
    selected_pretty = st.selectbox("Select Category:", ["Business", "Sci/Tech", "Sports", "World"], key="svm_feat_select")
    selected_raw = pretty_to_raw[selected_pretty]

    # Get class coefficients
    class_idx = list(svm.classes_).index(selected_raw)
    class_coef = svm.coef_[class_idx]

    # Get top 15 words and their coefficients
    feature_names = tfidf.get_feature_names_out()
    top_indices = class_coef.argsort()[-15:][::-1]

    top_df = pd.DataFrame({
        'Word': [feature_names[idx] for idx in top_indices],
        'Coefficient': [class_coef[idx] for idx in top_indices]
    }).sort_values(by='Coefficient', ascending=True)

    fig_imp = px.bar(
        top_df, x='Coefficient', y='Word',
        orientation='h',
        color_discrete_sequence=[topic_colors[selected_raw]]
    )
    fig_imp.update_layout(
        title=f"Top 15 Predictor Words for {selected_pretty}",
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    render_plotly_chart(fig_imp, use_container_width=True)
