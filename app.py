# -*- coding: utf-8 -*-
"""
Aplikasi Web untuk Klasifikasi Hoaks COVID-19 menggunakan Streamlit.
Aplikasi ini memungkinkan pengguna memasukkan judul berita dan
mengklasifikasikannya sebagai 'Hoaks' atau 'Nyata' menggunakan model ML yang telah dilatih.
"""

# ===== IMPOR LIBRARY =====
import streamlit as st # Library utama untuk membuat aplikasi web interaktif
import joblib # Untuk memuat model dan vektorizer yang telah disimpan
import re # Untuk operasi ekspresi reguler (pembersihan teks)
import string # Untuk string konstanta (pembersihan tanda baca)
from nltk.corpus import stopwords # Untuk daftar stop words
from nltk.tokenize import word_tokenize # Untuk tokenisasi teks
from nltk.stem import PorterStemmer # Untuk stemming kata
import nltk # Toolkit Natural Language
import pandas as pd # Untuk manipulasi data (meskipun terbatas di aplikasi ini)

# Mengunduh data NLTK yang diperlukan jika belum ada.
# Ini penting karena aplikasi Streamlit akan berjalan di lingkungan terpisah.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet') # Meskipun PorterStemmer digunakan, WordNet berguna.
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt_tab')
except nltk.downloader.DownloadError:
    nltk.download('punkt_tab')

# ===== DEFINISI KELAS PREPROSESOR TEKS =====
# Menggunakan kembali kelas TextPreprocessor yang sama persis dari skrip pelatihan
# untuk memastikan konsistensi dalam pra-pemrosesan teks input.
class TextPreprocessor:
    """
    Kelas untuk membersihkan dan memproses teks.
    Harus sama dengan kelas yang digunakan saat melatih model.
    """
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Membersihkan teks dari karakter yang tidak diinginkan."""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_stem(self, text):
        """Melakukan tokenisasi dan stemming teks."""
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def preprocess(self, text):
        """Pipa pra-pemrosesan lengkap."""
        text = self.clean_text(text)
        text = self.tokenize_and_stem(text)
        return text

# ===== MUAT MODEL DAN VEKTORIZER YANG TELAH DILATIH =====
# Bagian ini mencoba memuat objek-objek yang disimpan dari skrip pelatihan.
# Jika ada file yang tidak ditemukan, aplikasi akan menampilkan pesan kesalahan.
try:
    model = joblib.load('best_model.pkl') # Memuat model klasifikasi terbaik
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') # Memuat TF-IDF Vectorizer
    text_preprocessor = joblib.load('text_preprocessor.pkl') # Memuat objek preprocessor
    label_encoder = joblib.load('label_encoder.pkl') # Memuat label encoder
    st.success("Model, TF-IDF Vectorizer, Text Preprocessor, dan Label Encoder berhasil dimuat!")
except FileNotFoundError:
    st.error("""
        Error: File model atau vektorizer tidak ditemukan.
        Pastikan Anda telah menjalankan skrip pelatihan utama dan
        file 'best_model.pkl', 'tfidf_vectorizer.pkl', 'text_preprocessor.pkl', dan 'label_encoder.pkl'
        tersedia di direktori yang sama.
    """)
    st.stop() # Menghentikan eksekusi aplikasi jika file tidak ditemukan

# ===== ANTARMUKA PENGGUNA STREAMLIT =====
# Mengatur judul utama aplikasi
st.title("Klasifikasi Hoaks COVID-19")
st.markdown("---")

# Menambahkan deskripsi singkat aplikasi
st.write("""
Aplikasi ini menggunakan model Machine Learning (SGD, Logistic Regression, atau SVM)
yang dilatih untuk mengklasifikasikan judul berita terkait COVID-19 sebagai **Hoaks** atau **Nyata**.
Masukkan judul berita di bawah ini untuk mendapatkan prediksi.
""")

# Area input teks untuk pengguna
user_input = st.text_area("Masukkan Judul Berita COVID-19 di sini:", "")

# Tombol untuk melakukan klasifikasi
if st.button("Klasifikasi"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan judul berita untuk klasifikasi.")
    else:
        # Melakukan pra-pemrosesan teks input
        processed_input = text_preprocessor.preprocess(user_input)

        # Mengubah teks yang telah diproses menjadi representasi TF-IDF
        # Perlu array 1D karena `transform` mengharapkan input dalam bentuk list
        input_tfidf = tfidf_vectorizer.transform([processed_input])

        # Melakukan prediksi menggunakan model yang dimuat
        prediction_numeric = model.predict(input_tfidf)[0]

        # Mengonversi prediksi numerik kembali ke label string ('hoaks' atau 'nyata')
        prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]

        # Menampilkan hasil prediksi kepada pengguna
        st.markdown("---")
        st.subheader("Hasil Klasifikasi:")
        if prediction_label == 0: # Diasumsikan 0 = Hoaks, 1 = Nyata dari LabelEncoder
            st.error(f"**Judul Berita ini kemungkinan: HOAKS** 🚨")
        else:
            st.success(f"**Judul Berita ini kemungkinan: NYATA ✅**")

        st.write(f"Teks Asli: *{user_input}*")
        st.write(f"Teks Diproses: *{processed_input}*")

# Informasi tambahan atau footer
st.markdown("---")
st.info("Catatan: Model ini dilatih pada dataset spesifik dan kinerjanya dapat bervariasi dengan data baru.")
