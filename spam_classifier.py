import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Features and target
X = df['message']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Test with custom message
message = ["Congratulations! You've won a free iPhone. Click here to claim."]
message_vector = vectorizer.transform(message)

result = model.predict(message_vector)

if result[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: NOT SPAM")