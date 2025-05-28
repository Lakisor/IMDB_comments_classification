import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import joblib  # Saving vectorize model

train_df = pd.read_csv('data/processed/train_processed.csv')
texts = train_df['review'].values
labels = (train_df['sentiment'] == 'positive').astype(int).values

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
tfidf = TfidfVectorizer(max_features=10000)  # Vectorizer (limiting 10000 words)
X_train = tfidf.fit_transform(train_texts)
X_val = tfidf.transform(val_texts)

joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')  # Save this vectorizer for later use

X_train = torch.FloatTensor(X_train.toarray())
X_val = torch.FloatTensor(X_val.toarray())
y_train = torch.FloatTensor(train_labels).view(-1, 1)
y_val = torch.FloatTensor(val_labels).view(-1, 1)

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

input_size = X_train.shape[1]
model = SentimentNN(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
batch_size = 64

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean()
    return accuracy.item()

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_acc = evaluate(model, X_train, y_train)
    val_acc = evaluate(model, X_val, y_val)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

torch.save(model.state_dict(), 'models/sentiment_model.pth')
print("Training complete. Model saved to models/sentiment_model.pth")
