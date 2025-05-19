import streamlit as st
import joblib
import pandas as pd
import re
import string
import os

# Load model
model = joblib.load("model.pkl")

# Preprocessing function
def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# UI layout
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection")
st.write("Enter a news article below to check if it's Real or Fake.")

input_text = st.text_area("News Article", height=200)

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter a news article.")
    else:
        clean_text = wordpre(input_text)
        input_series = pd.Series([clean_text])
        prediction = model.predict(input_series)[0]
        label = "Real" if prediction == 1 else "Fake"

        st.success(f"### Prediction: This article is **{label}**.")

        # Feedback section
        st.write("---")
        st.write("### Was this prediction correct?")
        feedback = st.radio("Your feedback:", ["Yes", "No"], horizontal=True)

        if st.button("Submit Feedback"):
            feedback_data = pd.DataFrame([[input_text, label, feedback]],
                                         columns=["News_Text", "Prediction", "Feedback"])
            feedback_file = "U:/Projects/FND/data/feedback.csv"

            if os.path.exists(feedback_file):
                existing = pd.read_csv(feedback_file)
                feedback_data = pd.concat([existing, feedback_data], ignore_index=True)

            feedback_data.to_csv(feedback_file, index=False)
            st.success("Thank you for your feedback!")