import streamlit as st
import pandas as pd

def render_about_page():
    st.markdown("""
        <div class="header-banner">
            <h1>About Project</h1>
            <p>Detailed architecture overview, tools, and technical specifications</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<h3 style='color: #1e293b; font-weight: 600;'>Dataset Statistics & Mappings</h3>", unsafe_allow_html=True)
        st.write(
            "The project uses the **AG News Dataset**, one of the most widely used benchmark datasets "
            "for text classification research. Each sample consists of a news title and a short news description."
        )
        
        # Label Mappings Table
        mapping_df = pd.DataFrame({
            'Label': ['1', '2', '3', '4'],
            'Topic': ['World', 'Sports', 'Business', 'Sci/Tech']
        })
        st.dataframe(mapping_df, hide_index=True, use_container_width=True)
        
        # Dataset Quality & Stats
        st.write(
            "**Dataset Quality & Characteristics:**\n"
            "* **Total Articles:** 120,000 (perfectly balanced with 30,000 samples per class)\n"
            "* **Missing Values:** 0\n"
            "* **Duplicate Records:** 0\n\n"
            "The balanced nature of the dataset eliminates class imbalance issues and allows fair comparison "
            "between different classification algorithms."
        )
        
        # Dataset Sample Example Box
        st.markdown("""
            <div class="custom-card" style="background-color: #f8fafc; border-left: 4px solid #10b981; padding: 15px; margin-bottom: 25px;">
                <span style="font-weight: 600; color: #475569; font-size: 0.9rem;">Dataset Sample Example:</span><br>
                <div style="margin-top: 8px; font-size: 0.9rem;">
                    <b>Title:</b> Oil prices soar to all-time record<br>
                    <b>Description:</b> Global oil prices reached record highs amid concerns over supply shortages and geopolitical tensions.<br>
                    <b>Topic:</b> Business (Label 3)
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #1e293b; font-weight: 600;'>Data Preprocessing Pipeline</h3>", unsafe_allow_html=True)
        st.write(
            "Raw news articles often contain noisy information such as HTML remnants, encoded characters, URLs, "
            "and publisher-specific metadata. Before model training, an extensive text cleaning pipeline was performed:"
        )
        st.markdown("""
            1. Convert text to lowercase
            2. Remove URLs and web links
            3. Remove special characters and punctuation
            4. Remove Reuters and news source artifacts
            5. Tokenize text using spaCy
            6. Lemmatize words using spaCy
            7. Remove stopwords using NLTK
            8. Remove very short tokens and numerical noise
        """)
        
        # Carlyle Example Box
        st.markdown("""
            <div class="custom-card" style="background-color: #f8fafc; border-left: 4px solid #6366f1; padding: 15px; margin-bottom: 25px;">
                <span style="font-weight: 600; color: #475569; font-size: 0.9rem;">Preprocessing Example:</span><br>
                <div style="margin-top: 8px;">
                    <span style="font-weight: 500; font-size: 0.9rem; color: #64748b;">Original Text:</span>
                    <p style="font-size: 0.9rem; color: #0f172a; font-style: italic; margin: 2px 0 8px 0;">
                        "Reuters - Private investment firm Carlyle Group has shown interest in expanding its aerospace investments."
                    </p>
                    <span style="font-weight: 500; font-size: 0.9rem; color: #64748b;">Processed Text:</span>
                    <p style="font-size: 0.9rem; color: #4f46e5; font-weight: 500; margin: 2px 0 0 0;">
                        "private investment firm carlyle group show interest expand aerospace investment"
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #1e293b; font-weight: 600;'>Feature Engineering (TF-IDF)</h3>", unsafe_allow_html=True)
        st.write(
            "For traditional machine learning models, text was converted into numerical representations "
            "using TF-IDF (Term Frequency-Inverse Document Frequency) with the following configuration:"
        )
        st.markdown("""
            * **Maximum Features:** 20,000
            * **N-grams:** Unigrams + Bigrams (1,2)
            * **Vocabulary Filtering:** Applied
            
            TF-IDF captures the importance of words within a document while reducing the influence of commonly occurring terms.
        """)
        
    with col2:
        st.markdown("<h3 style='color: #1e293b; font-weight: 600; text-align: center;'>Model Architectures</h3>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="custom-card">
                <h5 style='color: #6366f1; margin:0 0 8px 0; font-weight: 600;'>Traditional Machine Learning</h5>
                <p style='color: #334155; font-size: 0.9rem; margin:0;'>
                    We trained three classical classifiers using high-dimensional sparse TF-IDF unigram and bigram features:
                </p>
                <ul style='font-size: 0.9rem; color: #334155; margin: 8px 0 0 0; padding-left: 20px;'>
                    <li><b>Multinomial Naive Bayes:</b> Probabilistic classifier.</li>
                    <li><b>Logistic Regression:</b> Linear classifier learning boundaries.</li>
                    <li><b>Linear Support Vector Machine (SVM):</b> Powerful margin-based classifier (best model).</li>
                </ul>
            </div>
            
            <div class="custom-card">
                <h5 style='color: #06b6d4; margin:0 0 8px 0; font-weight: 600;'>Deep Learning (BiLSTM)</h5>
                <p style='color: #334155; font-size: 0.9rem; margin:0;'>
                    A Bidirectional Long Short-Term Memory network implemented to capture contextual dependencies from both past and future words.
                </p>
                <div style='margin-top:8px; font-size:0.85rem; color:#475569;'>
                    <b>Architecture Flow:</b><br>
                    <code style='color:#0891b2;'>Input Text</code> → <code style='color:#0891b2;'>Embedding Layer</code> → <code style='color:#0891b2;'>Bidirectional LSTM</code> → <code style='color:#0891b2;'>Global Max Pooling</code> → <code style='color:#0891b2;'>Dense Layer</code> → <code style='color:#0891b2;'>Softmax Output</code>
                </div>
                <div style='margin-top:8px; font-size:0.85rem; color:#475569;'>
                    <b>Regularization:</b> Dropout, Recurrent Dropout, L2 Regularization, Early Stopping, Learning Rate Reduction.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #1e293b; font-weight: 600; text-align:center; margin-top:20px;'>Technological Stack</h3>", unsafe_allow_html=True)
        stack_df = pd.DataFrame({
            'Library / Tool': ['Streamlit', 'spaCy', 'NLTK', 'scikit-learn', 'Plotly', 'WordCloud'],
            'Purpose': ['Web UI & Interface', 'Lemmatization & NLP Parsing', 'Stopwords Corpus', 'ML Models & Vectorizer', 'Interactive Charts', 'Vocabulary Wordclouds']
        })
        st.dataframe(stack_df, hide_index=True, use_container_width=True)
        
        # Author details card
        st.markdown("""
            <div class="custom-card" style="margin-top: 20px; text-align: center; border-top: 4px solid #10b981;">
                <h5 style='margin:0; color: #10b981; font-weight: 700;'>Project Showcase</h5>
                <p style='margin:5px 0 0 0; color: #64748b; font-size: 0.95rem; font-weight: 500;'>Author: Varun Kaza</p>
                <p style='margin:2px 0 0 0; color: #64748b; font-size: 0.85rem;'>Academic Portfolio & Demonstration</p>
            </div>
        """, unsafe_allow_html=True)
