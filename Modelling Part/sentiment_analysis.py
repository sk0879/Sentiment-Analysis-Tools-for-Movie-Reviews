# sentiment_analysis.py
from textblob import TextBlob
import nltk

# Function to classify sentiment using TextBlob
def classify_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity  # Sentiment polarity (-1 to 1)
    
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Example usage:
if __name__ == "__main__":
    # Example movie review
    review = "I love this movie! The story was amazing and the characters were great."
    
    sentiment = classify_sentiment(review)
    print(f"Review sentiment: {sentiment}")
