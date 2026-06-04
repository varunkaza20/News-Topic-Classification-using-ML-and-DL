# News Topic Classification Using Machine Learning and Deep Learning

## Project Overview

News articles are generated at an enormous scale every day across digital platforms. Automatically categorizing news into predefined topics is an important Natural Language Processing (NLP) task that helps news aggregators, search engines, recommendation systems, and media organizations organize content efficiently.

This project develops an end-to-end News Topic Classification system capable of automatically assigning a news article to one of four categories:

* World
* Sports
* Business
* Sci/Tech

The project explores both traditional machine learning and deep learning approaches to understand how different techniques perform on short news articles. The complete workflow includes data preprocessing, exploratory data analysis, feature engineering, model training, performance evaluation, error analysis, and deployment through an interactive Streamlit application.

---

## Dataset

The project uses the AG News Dataset, one of the most widely used benchmark datasets for text classification research.

### Dataset Statistics

* Total Articles: 120,000
* Number of Classes: 4
* Balanced Dataset: 30,000 samples per class

### Categories

| Label | Topic    |
| ----- | -------- |
| 1     | World    |
| 2     | Sports   |
| 3     | Business |
| 4     | Sci/Tech |

Each sample consists of a news title and a short news description.

Example:

**Title:** Oil prices soar to all-time record

**Description:** Global oil prices reached record highs amid concerns over supply shortages and geopolitical tensions.

**Topic:** Business

The balanced nature of the dataset eliminates class imbalance issues and allows fair comparison between different classification algorithms.

---

## Data Preprocessing

Raw news articles often contain noisy information such as HTML remnants, encoded characters, URLs, and publisher-specific metadata. Before model training, extensive text cleaning was performed.

### Preprocessing Pipeline

1. Convert text to lowercase
2. Remove URLs and web links
3. Remove special characters and punctuation
4. Remove Reuters and news source artifacts
5. Tokenize text using spaCy
6. Lemmatize words using spaCy
7. Remove stopwords using NLTK
8. Remove very short tokens and numerical noise

### Example

**Original Text**

Reuters - Private investment firm Carlyle Group has shown interest in expanding its aerospace investments.

**Processed Text**

private investment firm carlyle group show interest expand aerospace investment

This preprocessing stage significantly reduces vocabulary noise while preserving meaningful semantic information.

---

## Exploratory Data Analysis (EDA)

A comprehensive exploratory analysis was performed to better understand the dataset characteristics.

### Dataset Quality

* Missing Values: 0
* Duplicate Records: 0
* Balanced Class Distribution

### Article Length Analysis

The cleaned dataset contains relatively short news descriptions.

#### Word Count Statistics

* Mean Length: ~19 words
* 95th Percentile: ~28 words
* 99th Percentile: ~39 words

These observations helped determine the optimal sequence length used later in the BiLSTM model.

### Character Count Analysis

Character-level statistics were analyzed to understand document complexity and distribution patterns.

### Vocabulary Analysis

The project examined:

* Most frequent words
* Category-wise frequent words
* Vocabulary size
* Word clouds
* N-gram distributions

Distinct vocabularies emerged across categories:

#### Sports

game, season, team, league, victory

#### Business

market, company, stock, profit, economy

#### World

government, president, country, minister, election

#### Sci/Tech

software, internet, technology, computer, microsoft

These patterns indicate that topic-specific language plays a significant role in classification performance.

---

## Feature Engineering

For traditional machine learning models, text was converted into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency).

### TF-IDF Configuration

* Maximum Features: 20,000
* N-grams: Unigrams + Bigrams
* Vocabulary Filtering: Applied

TF-IDF captures the importance of words within a document while reducing the influence of commonly occurring terms.

---

## Machine Learning Models

Three traditional machine learning models were trained and evaluated.

### Multinomial Naive Bayes

A probabilistic classifier commonly used for text classification tasks.

### Logistic Regression

A linear classification algorithm capable of learning decision boundaries using TF-IDF features.

### Linear Support Vector Machine (SVM)

A powerful margin-based classifier particularly effective for high-dimensional sparse text representations.

---

## Deep Learning Model

A Bidirectional Long Short-Term Memory (BiLSTM) network was implemented to capture contextual information from both past and future words in a sentence.

### Architecture

Input Text

↓

Embedding Layer

↓

Bidirectional LSTM

↓

Global Max Pooling

↓

Dense Layer

↓

Softmax Output Layer

### Regularization Techniques

* Dropout
* Recurrent Dropout
* L2 Regularization
* Early Stopping
* Learning Rate Reduction

These techniques were introduced to reduce overfitting and improve generalization performance.

---

## Model Evaluation

All models were evaluated using the same train-test split to ensure fair comparison.

### Performance Results

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | 89.74%   |
| Logistic Regression | 90.35%   |
| Linear SVM          | 90.49%   |
| BiLSTM              | ~90.0%   |

### Best Performing Model

Linear SVM achieved the highest classification accuracy of 90.49%.

Interestingly, the deep learning model did not significantly outperform traditional machine learning approaches despite its higher computational complexity.

---

## Error Analysis

To better understand model behavior, incorrectly classified samples were manually analyzed.

### Major Observations

#### Business ↔ Sci/Tech Confusion

The largest source of classification errors occurred between Business and Sci/Tech categories.

Examples include:

* Technology company earnings
* Product launches
* Telecommunications announcements
* Corporate acquisitions

These articles naturally contain vocabulary belonging to both categories.

#### World ↔ Business Confusion

Articles involving:

* International trade
* Economic agreements
* Energy markets
* Geopolitical developments

often overlapped between World and Business topics.

#### Sports Classification

Sports articles demonstrated the highest separability because of highly distinctive vocabulary and limited overlap with other categories.

---

## Key Findings

### Finding 1

Traditional machine learning models remain highly competitive for short-text classification tasks.

### Finding 2

Linear SVM achieved the best overall performance while requiring significantly less computational resources than BiLSTM.

### Finding 3

Business and Sci/Tech categories exhibit substantial semantic overlap, making them the most challenging classes to separate.

### Finding 4

The balanced nature of the dataset contributed to stable performance across all categories.

### Finding 5

For short news descriptions, keyword-based representations such as TF-IDF remain extremely effective.

---

## Real-Time Inference System

The final model was integrated into a real-time inference pipeline.

### Prediction Workflow

User Input

↓

Text Preprocessing

↓

TF-IDF Transformation

↓

Linear SVM Classification

↓

Predicted Topic

The system can classify unseen news articles into one of the four predefined categories in real time.

---

## Deployment

The project is deployed through a Streamlit web application featuring:

### Exploratory Data Analysis Dashboard

Interactive visualizations showing:

* Class distributions
* Word distributions
* Vocabulary statistics
* Word clouds
* Topic-wise analysis

### Real-Time News Classification

Users can paste any news article and instantly receive a predicted topic.

### Model Comparison Dashboard

Interactive comparison of:

* Accuracy
* Model performance
* Key findings
* Error analysis insights

---

## Conclusion

This project demonstrates a complete NLP workflow for automatic news categorization. Through extensive experimentation, traditional machine learning models, particularly Linear SVM combined with TF-IDF features, achieved the best balance between accuracy, efficiency, and interpretability. The results highlight that for short structured news articles, classical machine learning approaches remain highly effective and can compete with more complex deep learning architectures.
