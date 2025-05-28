import torch
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from train_model import SentimentNN

test_df = pd.read_csv('data/processed/test_processed.csv')
test_texts = test_df['review'].values
test_labels = (test_df['sentiment'] == 'positive').astype(int).values

model = SentimentNN(10000)
model.load_state_dict(torch.load('models/sentiment_model.pth'))
model.eval()

vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

X_test = vectorizer.transform(test_texts)
X_test = torch.FloatTensor(X_test.toarray())
y_test = torch.FloatTensor(test_labels).view(-1, 1)

with torch.no_grad():
    outputs = model(X_test)
    predictions = (outputs > 0.5).float()

accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['negative', 'positive']))

print("\nSample predictions:")
sample_indices = [0, 1, 2, 3, 4]
for idx in sample_indices:
    print(f"\nText: {test_texts[idx][:100]}...")
    print(f"True: {test_labels[idx]}, Predicted: {int(predictions[idx].item())} (Prob: {outputs[idx].item():.4f})")
