#1.A LOAD TOKENIZER
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.optim import AdamW  # ✅ Correct source now

#from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

#XXXXXXXXXXXXXXXXXXXXXXXXXXX
#1.B DATASET CLASS
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#XXXXXXXXXXXXXX
#1.C 
# Load and prepare data
fake = pd.read_csv("../data/Fake.csv")
real = pd.read_csv("../data/True.csv")
fake["label"] = 0
real["label"] = 1
df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["content"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

train_dataset = FakeNewsDataset(train_encodings, train_labels)
val_dataset = FakeNewsDataset(val_encodings, val_labels)

#XXXXXXXXXXXX
#1.D MODEL+TRAINER

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

#XXXXXXXXXXXX
#1.E TRAIN
trainer.train()

#XXXXXXXXXXXX
#1.F MODEL SAVED
model.save_pretrained("bert_model/model")
tokenizer.save_pretrained("bert_model/model")

print("BERT model trained and saved to bert_model/model/")
