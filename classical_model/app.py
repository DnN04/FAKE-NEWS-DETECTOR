import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

# Streamlit app
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter any news article content below to check if it's **FAKE** or **REAL**.")

# User input
user_input = st.text_area("Paste news content here:", height=200)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        
        if prediction == 0:
            st.error("🛑 This news is likely FAKE.")
        else:
            st.success("✅ This news is likely REAL.")
