from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer
import torch
from datasets import load_metric
from train_bert import val_dataset  # if you saved val_dataset

model = BertForSequenceClassification.from_pretrained('./bert_model/model')
tokenizer = BertTokenizer.from_pretrained('./bert_model/model')

trainer = Trainer(model=model)
metrics = trainer.evaluate(eval_dataset=val_dataset)
print(metrics)
