from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "bert_model"  # adjust path if needed
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Prediction function
def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return "REAL" if predicted_class == 1 else "FAKE"
