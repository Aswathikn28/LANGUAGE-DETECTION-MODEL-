
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import sklearn


print("scikit-learn version:", sklearn.__version__)  


df = pd.read_csv('langdetect.csv')  


def preprocess_text(text, language='English'):
    latin_languages = ['English', 'French', 'Spanish', 'Portugeese', 'Italian', 
                       'Sweedish', 'Dutch', 'German', 'Danish']
    if language in latin_languages:
        text = text.lower()
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['Text'] = df.apply(lambda row: preprocess_text(row['Text'], row['Language']), axis=1)
df = df.dropna()


tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Text']).toarray()
y = df['Language']


print("\nFeature matrix shape:", X.shape)  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


model = MultinomialNB()
model.fit(X_train, y_train)
print("\nModel training completed.")

joblib.dump(model, 'language_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully.")
