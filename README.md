# 📰 News Topic Classification using Machine Learning & Deep Learning

Deploy Link: [news-topic-classification-using-ml.streamlit.app](https://news-topic-classification-using-ml.streamlit.app/)

---

## 📖 About the Project

### Problem Statement
* News articles are generated at an enormous scale every day across digital platforms.
* Automatically categorizing news into predefined topics is an important Natural Language Processing (NLP) task.
* Categorization helps news aggregators, search engines, recommendation systems, and media organizations organize content efficiently.
* Short news descriptions and titles present unique challenges due to limited vocabulary size and context.

### Aim
* Automatically assign a news article to one of four categories: World, Sports, Business, or Sci/Tech.
* Explore both traditional machine learning (Naive Bayes, Logistic Regression, Linear SVM) and deep learning (BiLSTM) approaches on short news articles.

---

## 📊 Dataset Information

### Context
* The project uses the AG News Dataset, one of the most widely used benchmark datasets for text classification research.

### Content & Statistics
* **Total Articles:** 120,000
* **Number of Classes:** 4 (Balanced dataset: 30,000 samples per class)
* **Dataset Quality:** 0 missing values, 0 duplicate records
* **Word Count Statistics (Cleaned):** Mean length of ~19 words, 95th percentile of ~28 words, and 99th percentile of ~39 words.

| Label | Topic    |
| ----- | -------- |
| 1     | World    |
| 2     | Sports   |
| 3     | Business |
| 4     | Sci/Tech |

### Example
* **Title:** Oil prices soar to all-time record
* **Description:** Global oil prices reached record highs amid concerns over supply shortages and geopolitical tensions.
* **Topic:** Business

---

## ⚙️ Model Training & ML Techniques

1. **Data Preprocessing**:
   - Convert text to lowercase.
   - Remove URLs, web links, special characters, and punctuation.
   - Clean news source publisher artifacts (e.g., "Reuters - ").
   - Tokenize and lemmatize using `spaCy`.
   - Remove stopwords using `NLTK` and exclude numerical noise.
   * *Example:*
     * **Original:** `Reuters - Private investment firm Carlyle Group has shown interest in expanding its aerospace investments.`
     * **Processed:** `private investment firm carlyle group show interest expand aerospace investment`

2. **Feature Engineering**:
   - Numerical text representations generated using TF-IDF (Term Frequency-Inverse Document Frequency).
   - **TF-IDF Configuration:** Maximum 20,000 features, Unigrams + Bigrams, vocabulary filtering applied.

3. **Model Selection**:
   - Compared traditional probabilistic models (Multinomial Naive Bayes), linear boundaries (Logistic Regression), support vector classifiers (Linear SVM), and recurrent deep learning architectures (BiLSTM).
   - Deployed the **Linear SVM Classifier** for its optimal accuracy and low computational overhead.

4. **Deep Learning Architecture (BiLSTM)**:
   - Input Text → Embedding Layer → Bidirectional LSTM → Global Max Pooling → Dense Layer → Softmax.
   - Regularized using Dropout, Recurrent Dropout, L2 Regularization, Early Stopping, and Learning Rate Reduction.

---

## 🔍 Key Findings & Error Analysis

### Key Findings
* **Classical ML Efficiency:** Traditional machine learning models remain highly competitive for short-text classification tasks. Linear SVM achieved the best overall performance while requiring significantly fewer computational resources than BiLSTM.
* **Deep Learning Overfitting:** The BiLSTM model did not significantly outperform traditional machine learning approaches and was prone to overfitting under limited word sequence lengths.
* **Semantic Overlap:** Business and Sci/Tech categories exhibit substantial semantic overlap, making them the most challenging classes to separate.
* **Representation Power:** For short, structured news descriptions, keyword-based representations such as TF-IDF remain extremely effective.

### Error Analysis Insights
* **Business ↔ Sci/Tech Confusion:** The largest source of errors. Technology company earnings, product launches, telecommunications announcements, and corporate acquisitions naturally share vocabulary from both categories.
* **World ↔ Business Confusion:** Occurs frequently in articles covering international trade, economic agreements, energy markets, and geopolitical developments.
* **Sports Classification:** Demonstrated the highest separability because of highly distinctive, topic-specific vocabulary (e.g., game, season, team, league, victory) and limited overlap with other categories.

---

## 🛠️ Technology Stack

* **Language**: Python
* **Dashboard**: Streamlit (premium light-slate theme with Outfit typography)
* **ML/DL Libraries**: Scikit-learn, TensorFlow/Keras, spaCy, NLTK, Joblib/Pickle
* **Visuals**: Plotly, Matplotlib, WordCloud
* **Data**: Pandas, NumPy

---

## 📂 Project Structure

```
News-Topic-Classification/
├── app.py                      # Streamlit entry point
├── utils.py                    # Custom theme styling, metric cards, and data/model loaders
├── svm.pkl                     # Deployed Linear SVM model
├── tfidf.pkl                   # Trained TF-IDF Vectorizer
├── agnews_clean.csv            # Cleaned news dataset (used for EDA)
├── requirements.txt            # Package requirements with wheel paths for deployment
├── .gitignore                  # Git exclusion rules
├── .streamlit/
│   └── config.toml             # Custom theme settings and sidebar configurations
├── pages/
│   ├── home.py                 # Home/overview dashboard page
│   ├── eda.py                  # Exploratory Data Analysis page
│   ├── prediction.py           # Real-time topic prediction page
│   ├── comparison.py           # Model comparison and key findings page
│   └── about.py                # Tech stack & dataset metadata page
├── 1_Data_Analysis.ipynb       # [Ignored] Exploratory analysis notebook
└── 2_Modeling.ipynb           # [Ignored] Model training notebook
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/varunkaza20/News-Topic-Classification-using-ML-and-DL.git
cd News-Topic-Classification
```

### 2. Set up virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📊 Model Performance Comparison

| Model | Accuracy |
|-------|----------|
| **Linear SVM** *(deployed)* | **90.49%** |
| Logistic Regression | 90.35% |
| BiLSTM | ~90.00% |
| Multinomial Naive Bayes | 89.74% |

### Why Linear SVM was chosen for deployment:
* **Top Accuracy:** Reached the highest classification accuracy of 90.49% on the test split.
* **Low Latency:** SVM inference is extremely fast and executes in milliseconds on CPU, which is ideal for real-time web interface response times.
* **Simplicity and Efficiency:** Despite its architectural complexity and much longer training time, the BiLSTM model suffered from overfitting and did not outperform the linear classifier on short news descriptions.
