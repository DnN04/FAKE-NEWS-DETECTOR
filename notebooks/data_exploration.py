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
