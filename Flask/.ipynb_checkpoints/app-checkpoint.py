from flask import Flask, render_template, request
import joblib
import re
import string
import pandas as pd
import os
import csv

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
Model = joblib.load(model_path)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    original_text = None

    if request.method == 'POST':
        txt = request.form['txt']
        original_text = txt
        txt = wordpre(txt)
        txt = pd.Series([txt])

        try:
            result = Model.predict(txt)[0]
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, original_text=original_text)

@app.route('/feedback', methods=['POST'])
def feedback():
    news_text = request.form['original_text']
    prediction = request.form['prediction']
    feedback = request.form['feedback']

    with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([news_text, prediction, feedback])

    return render_template("index.html", result=None, original_text=None)

if __name__ == "__main__":
    app.run(debug=True)
