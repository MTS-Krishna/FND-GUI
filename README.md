# 📰 Fake News Detection using Machine Learning

This project is an intelligent web-based Fake News Detection System that classifies news content as either **real** or **fake** using state-of-the-art machine learning algorithms. It is designed to support real-time input and feedback via both **Flask** and **Streamlit** interfaces.

---

## 🚀 Project Overview

In an era dominated by information, fake news poses a serious threat to society. This system uses Natural Language Processing (NLP) and supervised machine learning models to detect misleading or false news articles based on their textual content.

---

## 📁 Project Structure

project_root/
│
├── FlaskApp/
│ ├── static/
│ ├── templates/
│ │ └── index.html
│ ├── feedback.csv
│ └── app.py
│
├── StreamlitApp/
│ └── streamlit_app.py
│
├── Model/
│ └── model.pkl
│
├── Dataset/
│ └── Dataset1.csv
│ └── ...
│
├── README.md
└── requirements.txt

yaml
Copy
Edit

---

## 🧠 Models Used

The following machine learning classifiers were trained and evaluated:

| Model                         | Accuracy (%) |
|------------------------------|--------------|
| Logistic Regression          | 87.14        |
| Stochastic Gradient Descent  | 86.32        |
| Random Forest                | 82.20        |
| Decision Tree                | 82.07        |
| Gradient Boosting            | 80.71        |
| XGBoost                      | 80.73        |
| Multinomial Naive Bayes      | 78.79        |
| Bernoulli Naive Bayes        | 76.08        |

🏆 **Logistic Regression** was selected for deployment based on performance.

---

## 📊 Dataset

- Multiple datasets were combined, including:
  - **LIAR dataset from PolitiFact (Kaggle)**
  - Additional datasets sourced and combined manually for variety.
- Dataset Type: **Textual**
- Total Samples: **~20,000+**
- Features: News content (Article)
- Target: **Binary (Fake = 1, Real = 0)**

---

## 🔧 Features

- End-to-End Data Preprocessing Pipeline
- Real-Time News Prediction
- Feedback Form for User Validation
- Dual Deployment Options:
  - Flask (HTML + CSS based)
  - Streamlit (Modern Python UI)

---

## 📦 Setup Instructions

### ✅ Prerequisites

Ensure Python 3.10+ is installed. Clone the repo:

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
