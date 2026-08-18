# Task 4: Sentiment Analysis on Movie Reviews

## Overview
This project implements a Natural Language Processing (NLP) pipeline to classify movie reviews as either positive or negative[cite: 1]. Using the NLTK `movie_reviews` corpus, text features are extracted via TF-IDF vectorization and classified using a Multinomial Naive Bayes model[cite: 1].

---

## Workflow

* **Data Extraction & Preprocessing:** Loaded the NLTK `movie_reviews` dataset, shuffled the documents, and combined tokenized words into full review texts[cite: 1].
* **Feature Extraction:** Transformed review text into numerical feature vectors using `TfidfVectorizer` (with English stop words removed)[cite: 1].
* **Model Training:** Split data into training (80%) and testing (20%) sets, then trained a `MultinomialNB` classifier[cite: 1].
* **Evaluation:** Assessed performance using accuracy, precision, recall, F1-score, and a confusion matrix heatmap[cite: 1].

---

## Model Evaluation

* **Overall Accuracy:** `82.5%`[cite: 1]

### Classification Report

| Sentiment | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Negative (`neg`)** | 0.82 | 0.84 | 0.83 | 202[cite: 1] |
| **Positive (`pos`)** | 0.83 | 0.81 | 0.82 | 198[cite: 1] |
| **Macro Average** | 0.83 | 0.82 | 0.82 | 400[cite: 1] |
| **Weighted Average** | 0.83 | 0.82 | 0.82 | 400[cite: 1] |

### Confusion Matrix Breakdown

* **True Negatives (TN):** 170[cite: 1]
* **False Positives (FP):** 32[cite: 1]
* **False Negatives (FN):** 38[cite: 1]
* **True Positives (TP):** 160[cite: 1]

---

## Key Insights

* The model demonstrates balanced performance across both classes, achieving an F1-score of 0.83 for negative reviews and 0.82 for positive reviews[cite: 1].
* Out of 400 test reviews, 330 were correctly identified while 70 were misclassified[cite: 1].
* TF-IDF paired with Multinomial Naive Bayes serves as an effective baseline for text-based sentiment classification[cite: 1].

---

## Tech Stack & Libraries

* **Language:** Python[cite: 1]
* **NLP & Text Processing:** NLTK, Scikit-Learn (`TfidfVectorizer`)[cite: 1]
* **Machine Learning:** Scikit-Learn (`MultinomialNB`)[cite: 1]
* **Data Manipulation & Visualization:** Pandas, Matplotlib, Seaborn[cite: 1]
