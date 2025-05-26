import torch
import torch.nn as nn
import joblib
import numpy as np
from preprocess_data import clean_text

# Define the same model architecture as in training
class SentimentNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        return self.sigmoid(self.layer3(x))

# Load the trained model and vectorizer
def load_model():
    model = SentimentNN(10000)  # Same as max_features in TF-IDF
    model.load_state_dict(torch.load('models/sentiment_model.pth'))
    model.eval()
    
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    # Clean and preprocess the input text
    cleaned_text = clean_text(text)
    
    # Transform text to TF-IDF features
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Convert to PyTorch tensor
    text_tensor = torch.FloatTensor(text_tfidf.toarray())
    
    # Get prediction
    with torch.no_grad():
        output = model(text_tensor)
        probability = output.item()
        
    # Convert to sentiment
    sentiment = 'positive' if probability > 0.5 else 'negative'
    confidence = probability if sentiment == 'positive' else 1 - probability
    
    return sentiment, confidence, probability

def main():
    print("Loading model and vectorizer...")
    model, vectorizer = load_model()
    print("Model loaded successfully!")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Enter your review (or 'quit' to exit): ")
        
        if text.lower() == 'quit':
            break
            
        if not text.strip():
            continue
            
        sentiment, confidence, prob = predict_sentiment(text, model, vectorizer)
        print(f"\nSentiment: {sentiment.upper()}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Raw probability: {prob:.4f}\n")

if __name__ == "__main__":
    main()
