# keyword_extraction.py
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract keywords
def extract_keywords(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Get the frequency distribution of keywords
    freq_dist = FreqDist(keywords)
    
    # Return the top 5 most common keywords
    return freq_dist.most_common(5)

# Example usage:
if __name__ == "__main__":
    review = "This is a wonderful movie with excellent acting and a gripping storyline."
    keywords = extract_keywords(review)
    print(f"Top keywords: {keywords}")
