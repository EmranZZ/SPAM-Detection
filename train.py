# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# 1) Load dataset
DATA_PATH = "data/SMSSpamCollection"
df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'text'], encoding='latin-1')
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("Total messages:", len(df))
print(df['label'].value_counts())

# Features & labels
X = df['text']
y = df['label_num']

# 2) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Build pipeline (TF-IDF + Naive Bayes)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)),
    ('clf', MultinomialNB())
])

# 4) Train
pipeline.fit(X_train, y_train)

# 5) Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6) Save model
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/spam_classifier.joblib')
print("Model saved to models/spam_classifier.joblib")
