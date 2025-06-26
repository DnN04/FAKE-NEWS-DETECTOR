#PHASE 1 - DATASET
import pandas as pd

# Load CSVs
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Add label columns
fake['label'] = 0
real['label'] = 1

# Combine datasets
df = pd.concat([fake, real], ignore_index=True)

# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# Print first few rows
print("First few rows:\n", df.head())

# ✅ Check missing values
print("\nMissing values:\n", df.isnull().sum())

# ✅ Check how many are fake vs real
print("\nLabel distribution:\n", df['label'].value_counts())


#PHASE 02.A- TEXT CLEANING
import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)
print(df['text'].head())


#PHASE 02.B- TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.7)  # ignore overly common words
X = tfidf.fit_transform(df['text'])  # transforms text to vector
y = df['label']

print(X.shape)  # check how many features we created
print("y example values:", y.head())


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#PHASE 3.A- SPLIT DATASET
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#PHASE 3.B- CLASSIFIER TRAINED
from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#PHASE 3.B- MODEL EVALUATION
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#PHASE 3.C- MODEL SAVING
import pickle

# Save model
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer (named 'tfidf' in your code)
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

sample = ["This is a completely fake news article with no truth."]
sample_vec = tfidf.transform(sample)
prediction = model.predict(sample_vec)
print("Prediction for sample:", "FAKE" if prediction[0] == 0 else "REAL")
