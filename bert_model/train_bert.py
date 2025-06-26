#1.APreprocess Data 
import pandas as pd

# Load and label
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")
fake["label"] = 0
real["label"] = 1

# Combine and shuffle
df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

# Combine title + text (BERT needs real sentences)
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

print(df.head())

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#1.B.TOKENIZE Preprocess Data 
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["content"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#1.C PYTORCH DATASET

import torch

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_dataset = FakeNewsDataset(train_encodings, train_labels)
val_dataset = FakeNewsDataset(val_encodings, val_labels)

#XXXXXXXXXXXXXXXXXXX

