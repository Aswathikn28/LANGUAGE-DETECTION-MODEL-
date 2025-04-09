
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import sklearn

# Confirm scikit-learn version
print("scikit-learn version:", sklearn.__version__)  # Should output 1.5.1

# Load the dataset
df = pd.read_csv('langdetect.csv')  # Adjust path if needed

# Define preprocessing function
def preprocess_text(text, language='English'):
    latin_languages = ['English', 'French', 'Spanish', 'Portugeese', 'Italian', 
                       'Sweedish', 'Dutch', 'German', 'Danish']
    if language in latin_languages:
        text = text.lower()
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocess the data
df['Text'] = df.apply(lambda row: preprocess_text(row['Text'], row['Language']), axis=1)
df = df.dropna()

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Text']).toarray()
y = df['Language']

# Check the shape
print("\nFeature matrix shape:", X.shape)  # Should be (10337, 5000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)
print("\nModel training completed.")

# Save the fitted model and vectorizer
joblib.dump(model, 'language_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully.")