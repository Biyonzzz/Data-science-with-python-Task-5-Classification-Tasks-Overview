# Data-science-with-python-Task-5-Classification-Tasks-Overview
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("C:/Users/bijay/Downloads/students.csv")


# Display first few rows
print("First 5 rows of dataset:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Define features and target variable
X = data[['Study_Hours', 'Attendance']]
y = data['Pass']  # Binary target: 0 = Fail, 1 = Pass

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# Uncomment to install dependencies
# !pip install nltk scikit-learn pandas numpy matplotlib seaborn

import nltk
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("C:/Users/bijay/Downloads/reviews.csv")  # Adjust path if needed

# Show first few rows
print("First 5 rows of dataset:")
print(df.head())

# Handle missing values
print("\nMissing Values:")
print(df.isnull().sum())
df.dropna(subset=['Review', 'Sentiment'], inplace=True)

# Encode sentiment if it's text-based
if df['Sentiment'].dtype == 'object':
    df['Sentiment'] = df['Sentiment'].str.lower().map({'positive': 1, 'negative': 0})

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Apply preprocessing
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

# Show sample cleaned data
print("\nSample Cleaned Data:")
print(df[['Review', 'Cleaned_Review']].head())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Cleaned_Review']).toarray()
y = df['Sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Sentiment Distribution
plt.figure(figsize=(5, 3))
df['Sentiment'].value_counts().plot(kind='bar', color=['salmon', 'skyblue'])
plt.title("Sentiment Distribution")
plt.xticks(ticks=[0,1], labels=["Negative", "Positive"], rotation=0)
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# OPTIONAL: Export predictions to CSV
results_df = pd.DataFrame({
    'Review': df['Review'].iloc[y_test.index],
    'Cleaned_Review': df['Cleaned_Review'].iloc[y_test.index],
    'Actual': y_test,
    'Predicted': y_pred
})
results_df.to_csv("predicted_reviews.csv", index=False)
print("\nPredictions exported to 'predicted_reviews.csv'")
