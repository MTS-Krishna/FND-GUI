# 📰 Fake News Detection Using Machine Learning

## Overview

Fake news has become a significant threat to digital ecosystems, political stability, and public trust. This project presents a machine learning-based system to classify news content as **real** or **fake** using advanced Natural Language Processing (NLP) techniques. It provides a clean, interactive deployment using both **Flask** and **Streamlit**, enabling real-time news verification.

---

## 🔍 Problem Statement

In an age where information spreads rapidly across digital platforms, fake news can manipulate public opinion and create widespread misinformation. Traditional methods of manual fact-checking are time-consuming and inefficient at scale. Hence, there is a need for an **automated, scalable, and intelligent fake news detection system** that leverages machine learning to analyze and classify news text based on learned patterns.

---

## 🎯 Objectives

- Build a system that can accurately classify news articles into fake or real categories.
- Explore and evaluate multiple machine learning models.
- Preprocess and clean raw textual data for optimal feature extraction.
- Provide an intuitive user interface for testing and verifying news articles.
- Log user feedback to improve future model performance.

---

## 📊 Dataset Description

- **Source**: Combined from **LIAR dataset (PolitiFact)** and four additional curated datasets.
- **Type**: Structured tabular dataset with text-based features.
- **Size**: ~20,000+ articles.
- **Features**: News content (headline + body), and a binary label (real/fake).
- **Target Variable**: `label` (0 for real, 1 for fake).
- **Nature**: Static, text-based dataset.

---

## 🧪 Model Summary

Eight different machine learning models were evaluated, including:

| Model                         | Accuracy (%) |
|------------------------------|--------------|
| Logistic Regression          | 87.14        |
| Stochastic Gradient Descent  | 86.32        |
| Random Forest Classifier     | 82.20        |
| Decision Tree Classifier     | 82.07        |
| Gradient Boosting Classifier | 80.71        |
| XGBoost Classifier           | 80.73        |
| Multinomial Naive Bayes      | 78.79        |
| Bernoulli Naive Bayes        | 76.08        |

🏆 **Logistic Regression** was selected for deployment due to its consistent and high performance.

---

## ⚙️ System Architecture

```
      ┌─────────────┐
      │  Raw News   │
      └─────┬───────┘
            │
     Text Preprocessing
            │
    Vectorization (TF-IDF)
            │
    Machine Learning Models
            │
  ┌─────────┴──────────┐
  │                    │
Prediction         Feedback Logging
  │                    │
  ▼                    ▼

```

---

## 🛠️ Tools and Technologies

- **Language**: Python 3.x
- **Libraries**:
  - Scikit-learn
  - XGBoost
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - Flask (Web Framework)
  - Streamlit (UI for ML)
- **Others**:
  - Joblib (for model serialization)
  - HTML/CSS (for Flask frontend)

---

## 🚀 Deployment Options

### Option 1: Flask Web App

```bash
cd Flask
python app.py
````

* Visit: `http://127.0.0.1:5000`
* Features a modern HTML/CSS interface
* Includes feedback logging (`feedback.csv`) to collect user validation

### Option 2: Streamlit App

```bash
cd Streamlit
streamlit run app.py
```

* Minimalist, responsive Python-native UI
* Live input, prediction, and logging interface

---

## 📂 Project Directory Structure

```
📁 Flask/
📁 Streamlit/
📁 dataset_1/
📁 dataset_2/
📁 dataset_3/
📁 dataset_4/
📁 dataset_5/
📄 Fakenewsdetection.ipynb
📄 README.md
📄 model.pkl
📄 requirements.txt
```

---

## 📬 Feedback System

The Flask and Streamlit applications both include a feedback section allowing users to indicate whether the model's prediction was correct or not. The system logs:

* Input news content
* Model prediction
* User feedback
  in `feedback.csv`, enabling future supervised fine-tuning or retraining.

---

## 🔮 Future Scope

* Integrate transformer-based models like **BERT**, **RoBERTa**, or **DistilBERT** for deeper contextual understanding.
* Enable real-time scraping and analysis of live news feeds from the web or social media.
* Implement active learning and self-training pipelines using user feedback.
* Expand language support to classify fake news in regional or non-English languages.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📌 Acknowledgments

* [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
* [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
* [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
* Open-source Python libraries and the ML research community
