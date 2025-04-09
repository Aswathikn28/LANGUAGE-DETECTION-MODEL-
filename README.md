#  Language Detection Web App

This is a web-based application that detects the **language** of any given text input using **Machine Learning (Naive Bayes)** and **TF-IDF vectorization**. The app is built using **Flask** for the backend and styled using **HTML/CSS** for a clean and responsive user interface.

---

##  Features

- Detects language from user input text
- Trained on a multilingual dataset
- Uses **TF-IDF** vectorization for feature extraction
- **Naive Bayes Classifier** for prediction
- Flask-based web interface
- Clean and animated UI
- Responsive design (mobile-friendly)
- Supports the following languages:
  - English
  - Hindi
  - French
  - Spanish
  - Portuguese
  - Russian
  - Italian
  - Swedish
  - Dutch
  - Kannada
  - Malayalam
  - Tamil
  - Turkish
  - Arabic
  - Danish
  - German
  - Greek

---

##  Machine Learning Model

- **Dataset**: `langdetect.csv` (text and language columns)
- **Preprocessing**:
  - Text lowercasing
  - Removing digits and punctuation
  - Removing extra whitespace
- **Model**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF (`max_features=5000`)
- **Training/Test Split**: 80/20
- **Model Persistence**: `joblib` used to save the model and vectorizer

---

##  Tech Stack

- Python
- Scikit-learn
- Pandas
- Flask
- HTML/CSS
- Joblib

---

##  Project Structure

```bash
language-detector/
│
├── templates/
│   └── index.html          # Frontend template
├── static/
│   └── bg2.jpeg            # Background image
├── langdetect.csv          # Dataset (not included here)
├── app.py                  # Flask web app
├── train.py          # Script to train and save the model
├── language_model.pkl      # Trained model
├── tfidf_vectorizer.pkl    # Saved vectorizer
└── README.md               # This file
