# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

# Load dataset (Assume you have a CSV with 'review' and 'sentiment' columns)
df = pd.read_csv('data/sample_reviews.csv')

# Preprocess the reviews (you can use the 'preprocess_text' function from preprocess.py)
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2)

# Build a machine learning pipeline (TF-IDF + Naive Bayes)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
with open('models/sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Evaluate the model (optional)
print(f"Model accuracy: {model.score(X_test, y_test)}")
