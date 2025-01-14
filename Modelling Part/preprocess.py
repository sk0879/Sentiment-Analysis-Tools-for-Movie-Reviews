# preprocess.py
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Preprocess movie review
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Return cleaned text
    return ' '.join(filtered_words)

# Example usage:
if __name__ == "__main__":
    review = "This is a great movie, with fantastic acting and a brilliant plot!"
    clean_review = preprocess_text(review)
    print(f"Cleaned review: {clean_review}")
