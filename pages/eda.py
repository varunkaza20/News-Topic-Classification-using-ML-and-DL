import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from utils import render_metric_card, topic_prettymap, topic_colors, render_plotly_chart

def render_eda_page(df):
    st.markdown("""
        <div class="header-banner">
            <h1>Exploratory Data Analysis</h1>
            <p>Analyze class balance, distributions, word counts, and topic-specific vocabulary statistics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Section 6: Vocabulary Statistics (KPI Cards at the top)
    st.markdown("<h3 style='color: #1e293b; font-weight: 600; margin-bottom: 15px;'>Vocabulary & Dataset Stats</h3>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card("Total Vocab Size", "46,737", "Unique words in corpus")
    with kpi2:
        render_metric_card("Avg Words/Article", "19.20", "Words per description")
    with kpi3:
        render_metric_card("Avg Chars/Article", "131.59", "Characters per article")
    with kpi4:
        render_metric_card("95th %ile Word Count", "28", "Words per article limit")
        
    st.markdown("<hr style='border-top: 1px solid #cbd5e1; margin: 30px 0;'>", unsafe_allow_html=True)

    # 1. Class Distribution Plot
    st.markdown("### 1. Class Distribution")
    
    counts = df['topic'].value_counts().reset_index()
    counts.columns = ['Topic', 'Count']
    counts['Topic'] = counts['Topic'].map(topic_prettymap)
    fig1 = px.bar(
        counts, x='Topic', y='Count',
        color='Topic',
        color_discrete_map={topic_prettymap[k]: v for k, v in topic_colors.items()},
        title="Class Counts (Balanced Train Split)"
    )
    fig1.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    render_plotly_chart(fig1, use_container_width=True)
    
    st.markdown("""
        **insights:**
        * The dataset is perfectly balanced with exactly **30,000 articles per class** (120,000 total).
        * This balanced nature of the dataset eliminates class imbalance issues and allows fair comparison between different classification algorithms.
    """)
    st.markdown("<hr style='border-top: 1px dashed #cbd5e1; margin: 30px 0;'>", unsafe_allow_html=True)

    # 2. Word Count Distribution
    st.markdown("### 2. Word Count Distribution")
    
    fig2 = px.histogram(
        df, x='word_count', nbins=50,
        title="Word Count Histogram",
        color_discrete_sequence=['#6366f1']
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    render_plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
        **insights:**
        * Most news article descriptions are relatively short, peaking around **15–25 words per article**.
        * The distribution is slightly right-skewed. The average word count is **~19 words**, the 95th percentile is **~28 words**, and the 99th percentile is **~39 words**.
        * These observations helped determine the optimal sequence length used later in the BiLSTM deep learning model.
    """)
    st.markdown("<hr style='border-top: 1px dashed #cbd5e1; margin: 30px 0;'>", unsafe_allow_html=True)

    # 3. Character Count Distribution
    st.markdown("### 3. Character Count Distribution")
    
    fig3 = px.histogram(
        df, x='char_count', nbins=50,
        title="Character Count Histogram",
        color_discrete_sequence=['#06b6d4']
    )
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    render_plotly_chart(fig3, use_container_width=True)
    
    st.markdown("""
        **insights:**
        * Character-level statistics were analyzed to understand document complexity and distribution patterns.
        * Most descriptions peak between **100 and 150 characters**, which confirms a consistent average word length and complexity across all classes.
    """)
    st.markdown("<hr style='border-top: 1px dashed #cbd5e1; margin: 30px 0;'>", unsafe_allow_html=True)

    # 4. Word Clouds
    st.markdown("### 4. Word Clouds by Topic")
    
    @st.cache_data
    def generate_wordclouds():
        clouds = {}
        for t in ['world', 'sports', 'business', 'sci_tech']:
            t_text = ' '.join(df[df['topic'] == t]['clean_text'].dropna().sample(5000, random_state=42))
            wc = WordCloud(width=400, height=300, background_color='white', colormap='cool').generate(t_text)
            clouds[t] = wc
        return clouds
        
    clouds = generate_wordclouds()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h5 style='text-align: center; color:#06b6d4;'>World</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.imshow(clouds['world'], interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        st.markdown("<h5 style='text-align: center; color:#f59e0b;'>Business</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.imshow(clouds['business'], interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
    with col2:
        st.markdown("<h5 style='text-align: center; color:#10b981;'>Sports</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.imshow(clouds['sports'], interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        st.markdown("<h5 style='text-align: center; color:#6366f1;'>Sci/Tech</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.imshow(clouds['sci_tech'], interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
    st.markdown("""
        **insights:**
        * Distinct topic-specific vocabularies emerge across each of the four categories:
            * **World:** Frequently contains terms like `government`, `president`, `country`, `minister`, `election`.
            * **Sports:** Focuses heavily on team and action terms like `game`, `season`, `team`, `league`, `victory`.
            * **Business:** Features corporate and financial terms like `market`, `company`, `stock`, `profit`, `economy`.
            * **Sci/Tech:** Highlighted by words like `software`, `internet`, `technology`, `computer`, `microsoft`.
    """)
    st.markdown("<hr style='border-top: 1px dashed #cbd5e1; margin: 30px 0;'>", unsafe_allow_html=True)    # 5. Top Words by Topic
    st.markdown("### 5. Top Words by Topic")
    
    @st.cache_data
    def get_top_terms(topic_name, n=15):
        df_topic = df[df['topic'] == topic_name]
        all_words = ' '.join(df_topic['clean_text'].dropna().astype(str)).lower().split()
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(n)
        return pd.DataFrame(top_words, columns=['Word', 'Count'])
 
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h5 style='color:#06b6d4;'>World Top Words</h5>", unsafe_allow_html=True)
        top_w = get_top_terms('world')
        fig = px.bar(top_w, x='Count', y='Word', orientation='h', color_discrete_sequence=['#06b6d4'])
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        render_plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h5 style='color:#f59e0b;'>Business Top Words</h5>", unsafe_allow_html=True)
        top_w = get_top_terms('business')
        fig = px.bar(top_w, x='Count', y='Word', orientation='h', color_discrete_sequence=['#f59e0b'])
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        render_plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("<h5 style='color:#10b981;'>Sports Top Words</h5>", unsafe_allow_html=True)
        top_w = get_top_terms('sports')
        fig = px.bar(top_w, x='Count', y='Word', orientation='h', color_discrete_sequence=['#10b981'])
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        render_plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h5 style='color:#6366f1;'>Sci/Tech Top Words</h5>", unsafe_allow_html=True)
        top_w = get_top_terms('sci_tech')
        fig = px.bar(top_w, x='Count', y='Word', orientation='h', color_discrete_sequence=['#6366f1'])
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        render_plotly_chart(fig, use_container_width=True)
        
    st.markdown("""
        **insights:**
        * Topic-specific language plays a significant role in classification performance.
        * Terms like `new`, `company`, and `said` are highly frequent but are shared across multiple classes, which is why TF-IDF scaling is critical.
        * TF-IDF downweights these commonly occurring terms and highlights highly specific keywords, which helps models achieve stable and high classification accuracy.
    """)
