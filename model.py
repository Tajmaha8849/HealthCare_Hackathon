import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import joblib

nltk.download('punkt')
nltk.download('stopwords')

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load and preprocess the dataset
data = pd.read_csv('data_disease.csv')
data['processed_text'] = data['Symptoms'].apply(lambda x: preprocess_text(x))

# Train the model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['processed_text'])
y = data['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model,"ChatBot.pkl")
