import re
from flask import Flask, request, render_template
import joblib
import numpy as np
import sklearn


app = Flask(__name__)


print("scikit-learn version:", sklearn.__version__) 


try:
    model = joblib.load('language_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    exit()

 
def preprocess_text(text, language='English'):
    latin_languages = ['English', 'French', 'Spanish', 'Portugeese', 'Italian', 
                       'Sweedish', 'Dutch', 'German', 'Danish']
    if language in latin_languages:
        text = text.lower()
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_language(text):
    processed_text = preprocess_text(text, 'English')
    text_vector = tfidf.transform([processed_text]).toarray()
    prediction = model.predict(text_vector)
    return prediction[0]


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    input_text = None
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text:
            try:
                prediction = predict_language(input_text)
            except Exception as e:
                prediction = f"Error: {str(e)}"
    return render_template('index.html', prediction=prediction, input_text=input_text)


if __name__ == '__main__':
    app.run(debug=True)
