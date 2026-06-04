import streamlit as st
import pandas as pd
import plotly.express as px
from utils import preprocess_text, topic_prettymap, topic_colors, render_plotly_chart

def render_prediction_page(svm, tfidf, nlp):
    st.markdown("""
        <div class="header-banner">
            <h1>Real-Time Topic Prediction</h1>
            <p>Enter text or select a sample article to classify with the trained Linear SVM model</p>
        </div>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 25px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-weight: 600; color: #475569; font-size: 0.95rem; margin-bottom: 8px;">Real-Time Inference Pipeline:</div>
            <span style="color: #6366f1; font-weight: 600; font-size: 0.95rem;">User Input</span> → 
            <span style="color: #6366f1; font-weight: 600; font-size: 0.95rem;">Text Preprocessing</span> → 
            <span style="color: #6366f1; font-weight: 600; font-size: 0.95rem;">TF-IDF Transformation</span> → 
            <span style="color: #6366f1; font-weight: 600; font-size: 0.95rem;">Linear SVM Classification</span> → 
            <span style="color: #6366f1; font-weight: 600; font-size: 0.95rem;">Predicted Topic</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Session state for user inputs
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    # Sample News Buttons
    st.markdown("<h5 style='color: #334155; font-weight: 600; margin-bottom: 10px;'>Quick Test Templates</h5>", unsafe_allow_html=True)
    
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("💼 Business Example"):
            st.session_state.text_input = "Apple shares rose after the company announced strong quarterly earnings."
    with btn_col2:
        if st.button("⚽ Sports Example"):
            st.session_state.text_input = "Manchester United won the match after scoring two late goals."
    with btn_col3:
        if st.button("🌍 World Example"):
            st.session_state.text_input = "World leaders gathered to discuss climate policy and international cooperation."
    with btn_col4:
        if st.button("🚀 Sci/Tech Example"):
            st.session_state.text_input = "Google unveiled a new artificial intelligence model for software development."

    # Text Input Area
    user_text = st.text_area(
        "News Article Text:",
        value=st.session_state.text_input,
        placeholder="Paste a news article here...",
        height=180
    )
    
    predict_btn = st.button("Predict Topic")
    
    if predict_btn and user_text.strip():
        with st.spinner("Analyzing and preprocessing text..."):
            cleaned = preprocess_text(user_text, nlp)
            features = tfidf.transform([cleaned])
            pred_class = svm.predict(features)[0]
            dec_scores = svm.decision_function(features)[0]
            
            pretty_pred = topic_prettymap[pred_class]
            theme_class = f"{pred_class.replace('_','')}-theme"
            
            st.markdown(f"""
                <div class="prediction-card {theme_class}">
                    <h3 style='margin:0; font-size: 1.8rem; font-weight: 700; color: white;'>Predicted Topic: {pretty_pred}</h3>
                    <p style='margin: 8px 0 0 0; font-size: 0.95rem; opacity: 0.9;'>Prediction generated using trained Linear SVM model</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h4 style='color: #1e293b; font-weight: 600;'>Model Decision Scores</h4>", unsafe_allow_html=True)
            st.write("Linear SVM boundary margin scores for each class (higher is more confident):")
            
            pretty_classes = [topic_prettymap[c] for c in svm.classes_]
            decision_df = pd.DataFrame({
                'Topic': pretty_classes,
                'Decision Score': dec_scores
            }).sort_values(by='Decision Score', ascending=True)
            
            fig = px.bar(
                decision_df, x='Decision Score', y='Topic',
                orientation='h',
                color='Topic',
                color_discrete_map={topic_prettymap[k]: v for k, v in topic_colors.items()},
                title="Support Vector Boundary Distance"
            )
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            render_plotly_chart(fig, use_container_width=True)
