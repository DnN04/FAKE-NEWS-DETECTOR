# import streamlit as st
# from bert_predict import predict_news

# from transformers import BertTokenizer, BertForSequenceClassification
# import torch

# MODEL_PATH = "./bert_model/model"  # Path to your saved model folder

# # Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# model = BertForSequenceClassification.from_pretrained(MODEL_PATH)


# st.set_page_config(page_title="BERT Fake News Detector", layout="centered")

# st.title("🧠 BERT Fake News Detector")
# user_input = st.text_area("Enter the news content to verify:")

# if st.button("Check"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         prediction = predict_news(user_input)
#         if prediction == "REAL":
#             st.success("✅ This news is likely REAL.")
#         else:
#             st.error("⚠️ This news is likely FAKE.")



# --- IMPORTS ---
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# --- LOAD MODEL ---
MODEL_PATH = "./bert_model/model"  # Adjust if needed
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


# Streamlit UI
st.title("Fake News Detector - BERT")


user_input = st.text_area("Enter a news article")

if st.button("Predict"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

        st.write(f"🧠 **Confidence Scores:**")
        st.write(f"- Fake: `{fake_prob:.2f}`")
        st.write(f"- Real: `{real_prob:.2f}`")

        if pred == 0:
            st.error("🚫 The news is predicted to be **FAKE**.")
        else:
            st.success("✅ The news is predicted to be **REAL**.")

