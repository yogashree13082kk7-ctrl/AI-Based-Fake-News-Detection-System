import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
data = pd.read_csv(BASE_DIR / "news.csv")

X = data["text"].astype(str)
y = data["label"].astype(str).str.upper()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate model
predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy * 100, "%")
print(classification_report(y_test, predictions))

# Save model and vectorizer
joblib.dump(model, BASE_DIR / "model.pkl")
joblib.dump(vectorizer, BASE_DIR / "vectorizer.pkl")

print("Model trained successfully!")
print("model.pkl and vectorizer.pkl created.")
