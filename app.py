# app.py

# ===== INSTALASI DEPENDENSI =====
# Pastikan Anda telah menginstal Streamlit dan dependensi lainnya.
# !pip install streamlit joblib pandas scikit-learn nltk

import streamlit as st
import pandas as pd
import joblib
from nltk.tokenize import word_tokenize
import re
import string

# ===== MUAT MODEL DAN PREPROCESSOR =====
# Memuat model terbaik, TF-IDF vectorizer, dan preprocessor
best_model = joblib.load('best_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
text_preprocessor = joblib.load('text_preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ===== FUNKSI UNTUK MEMPROSES TEKS =====
def clean_text(text):
    """
    Membersihkan teks dari karakter yang tidak diinginkan seperti URL,
    mention, hashtag, angka, dan tanda baca. Mengubah teks menjadi huruf kecil
    dan menghapus spasi ekstra.
    """
    if pd.isna(text):  # Menangani nilai NaN (Not a Number)
        return ""

    # Mengonversi teks ke huruf kecil
    text = text.lower()

    # Menghapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Menghapus mention (@username) dan hashtag (#tag)
    text = re.sub(r'@\w+|#\w+', '', text)

    # Menghapus angka
    text = re.sub(r'\d+', '', text)

    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Menghapus spasi ekstra dan memangkas
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_text(text):
    """
    Melakukan pra-pemrosesan pada teks.
    """
    text = clean_text(text)  # Langkah pembersihan
    tokens = word_tokenize(text)  # Tokenisasi
    return ' '.join(tokens)

# ===== ANTARMUKA STREAMLIT =====
st.title("Klasifikasi Hoaks COVID-19")
st.write("Masukkan berita COVID-19 yang ingin Anda klasifikasikan:")

# Input teks dari pengguna
user_input = st.text_area("Berita COVID-19:", height=150)

if st.button("Klasifikasikan"):
    if user_input:
        # Pra-pemrosesan teks
        processed_text = preprocess_text(user_input)

        # Mengubah teks menjadi matriks TF-IDF
        text_tfidf = tfidf_vectorizer.transform([processed_text])

        # Melakukan prediksi
        prediction = best_model.predict(text_tfidf)
        prediction_proba = best_model.predict_proba(text_tfidf)

        # Mengonversi prediksi ke label asli
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = prediction_proba[0][1] if predicted_label == 'nyata' else prediction_proba[0][0]

        # Menampilkan hasil
        st.write(f"**Prediksi:** {predicted_label}")
        st.write(f"**Tingkat Keyakinan:** {confidence:.2f}")
    else:
        st.warning("Silakan masukkan berita untuk diklasifikasikan.")

# ===== MENJALANKAN APLIKASI =====
if __name__ == "__main__":
    st.write("Aplikasi ini menggunakan model klasifikasi hoaks COVID-19.")
