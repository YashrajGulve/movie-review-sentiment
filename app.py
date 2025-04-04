import streamlit as st
import joblib
import re
import string

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    return text

# Streamlit App UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below, and the AI will predict if it's Positive or Negative!")

# Text input
review_input = st.text_area("Enter your movie review here:")

if st.button("Analyze Sentiment"):
    if review_input.strip():
        cleaned_review = clean_text(review_input)
        vectorized_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_review)[0]
        st.success(f"🎭 Sentiment: **{prediction.capitalize()}**")
    else:
        st.warning("Please enter a review before clicking the button.")
