import os
import re
import string
import pandas as pd
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    
    text = re.sub(r'<[^>]+>', ' ', text)
    
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    text = re.sub(r'\b\d+\b', ' NUMBER ', text)
    
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    tokens = word_tokenize(text)
    
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 1]
    
    tokens = [token for token in tokens if len(token) > 1 or token in ['i', 'a']]
    
    text = ' '.join(tokens)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data():
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.read_csv("data/raw/train.csv", quotechar='"', quoting=1)
    train_df['review'] = train_df['review'].apply(clean_text)
    train_df.to_csv(processed_dir / "train_processed.csv", index=False, quotechar='"', quoting=1)
    
    test_df = pd.read_csv("data/raw/test.csv", quotechar='"', quoting=1)
    test_df['review'] = test_df['review'].apply(clean_text)
    test_df.to_csv(processed_dir / "test_processed.csv", index=False, quotechar='"', quoting=1)

if __name__ == "__main__":
    preprocess_data()
    print("Data preprocessing completed. Processed files saved to data/processed/")
