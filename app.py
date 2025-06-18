import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib # To save/load the trained model and vectorizer

# --- Load Model and Vectorizer ---
# You'll need to save your trained SGD model and TF-IDF vectorizer first.
# Run these lines in your original script after training:
# joblib.dump(sgd, 'sgd_model.pkl')
# joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

try:
    sgd_model = joblib.load('sgd_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure 'sgd_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()

# --- NLTK Downloads (only if not already downloaded) ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Text Preprocessing Function (from your original script) ---
stemmer = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]',' ',content)
    content = content.lower()
    content = content.split()
    content = [stemmer.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content

# --- Streamlit App ---
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")

st.markdown("""
    This application uses a pre-trained **SGD Classifier** model
    to predict whether a news headline is **fake** or **true**.
    Just enter a headline in the text area below and click 'Predict'!
""")

st.header("Enter News Headline")
user_input = st.text_area("Paste your news headline here:", height=150, placeholder="Type or paste a news headline...")

if st.button("Predict"):
    if user_input:
        with st.spinner("Analyzing..."):
            # 1. Preprocess the input
            preprocessed_input = stemming(user_input)

            # 2. Transform the preprocessed input using the loaded TF-IDF vectorizer
            vectorized_input = tfidf_vectorizer.transform([preprocessed_input])

            # 3. Make prediction
            prediction = sgd_model.predict(vectorized_input)
            prediction_proba = sgd_model.predict_proba(vectorized_input)

            st.subheader("Prediction Result:")
            if prediction[0] == 0:
                st.error("🚨 This news is likely **FAKE**.")
            else:
                st.success("✅ This news is likely **TRUE**.")

            st.markdown(f"**Confidence (Fake vs. True):**")
            st.write(f"Fake: {prediction_proba[0][0]*100:.2f}%")
            st.write(f"True: {prediction_proba[0][1]*100:.2f}%")

            st.info("💡 **How it works:** The model analyzes the textual patterns in your headline based on what it learned from a large dataset of fake and real news. It then provides a prediction along with a confidence score.")

    else:
        st.warning("Please enter a news headline to get a prediction.")

st.markdown("---")
st.markdown("Created with ❤️ by your AI Assistant")
