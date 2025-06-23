import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model components
try:
    best_model = joblib.load('best_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    text_preprocessor = joblib.load('text_preprocessor.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    st.success("Model components loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model components not found. Please ensure 'best_model.pkl', 'tfidf_vectorizer.pkl', 'text_preprocessor.pkl', and 'label_encoder.pkl' are in the same directory.")
    st.stop() # Stop the app if essential files are missing

# --- Streamlit App Layout ---
st.set_page_config(page_title="COVID-19 Hoax Classifier", layout="centered")

st.title("🦠 COVID-19 Hoax Classifier")

st.markdown("""
    This application classifies headlines related to COVID-19 as either **Hoax** or **Real**.
    Enter a headline in the text area below and click 'Predict' to see the classification.
""")

# Input text area for user
user_input = st.text_area("Enter a headline here:", height=150, placeholder="e.g., Drinking bleach cures COVID-19")

if st.button("Predict"):
    if user_input:
        # 1. Preprocess the input text
        with st.spinner("Processing text..."):
            processed_input = text_preprocessor.preprocess(user_input)

        # 2. Transform the processed text using the loaded TF-IDF vectorizer
        # We need to reshape for single sample prediction
        input_tfidf = tfidf_vectorizer.transform([processed_input])

        # 3. Make prediction
        prediction_label_encoded = best_model.predict(input_tfidf)[0]
        prediction_text = label_encoder.inverse_transform([prediction_label_encoded])[0] # Convert back to 'hoax' or 'nyata'

        # 4. Get prediction probability (if available)
        if hasattr(best_model, 'predict_proba'):
            # Get probability for the predicted class
            prediction_proba = best_model.predict_proba(input_tfidf)[0][prediction_label_encoded]
            confidence = prediction_proba * 100
        else:
            confidence = None
            st.warning("The selected model does not support probability prediction.")

        st.markdown("---")
        st.subheader("Prediction Result:")

        if prediction_text == 'hoaks':
            st.error(f"This headline is classified as: **HOAX**")
            if confidence is not None:
                st.write(f"Confidence: **{confidence:.2f}%**")
        else:
            st.success(f"This headline is classified as: **REAL**")
            if confidence is not None:
                st.write(f"Confidence: **{confidence:.2f}%**")

        st.markdown(f"**Processed Text:** `{processed_input}`")
        
        st.markdown("---")
        st.info("Disclaimer: This model is for demonstrative purposes and should not be used as a sole source for verifying information. Always consult official health organizations for accurate information.")

    else:
        st.warning("Please enter some text to get a prediction.")

st.markdown("""
    <style>
    .reportview-container .main .block-container{
        max-width: 800px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("About")
st.sidebar.info(
    "This application uses a pre-trained machine learning model to classify COVID-19 related "
    "headlines. The model was trained using TF-IDF features and a **Stochastic Gradient Descent (SGD)** "
    "classifier (or Logistic Regression/SVM, depending on which was best)."
)

st.sidebar.header("How it Works")
st.sidebar.markdown("""
- **Text Preprocessing:** The input text is cleaned (lowercase, remove URLs, punctuation, numbers) and then tokenized and stemmed.
- **Feature Extraction (TF-IDF):** The preprocessed text is converted into numerical features using TF-IDF.
- **Classification:** The trained model then predicts whether the headline is a 'Hoax' or 'Real'.
""")

st.sidebar.header("Model Details")
if 'best_model' in locals():
    st.sidebar.write(f"**Best Model Used:** {type(best_model).__name__}")
    if hasattr(tfidf_vectorizer, 'max_features'):
        st.sidebar.write(f"**TF-IDF Max Features:** {tfidf_vectorizer.max_features}")
    if hasattr(tfidf_vectorizer, 'ngram_range'):
        st.sidebar.write(f"**TF-IDF Ngram Range:** {tfidf_vectorizer.ngram_range}")
