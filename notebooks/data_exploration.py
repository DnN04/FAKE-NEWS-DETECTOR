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

