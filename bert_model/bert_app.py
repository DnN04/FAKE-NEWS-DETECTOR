import streamlit as st
from bert_predict import predict_news

st.set_page_config(page_title="BERT Fake News Detector", layout="centered")

st.title("🧠 BERT Fake News Detector")
user_input = st.text_area("Enter the news content to verify:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = predict_news(user_input)
        if prediction == "REAL":
            st.success("✅ This news is likely REAL.")
        else:
            st.error("⚠️ This news is likely FAKE.")
