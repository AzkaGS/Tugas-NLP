import streamlit as st
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="COVID-19 Hoax Detection",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Custom CSS (to mimic the original design) ---
st.markdown("""
<style>
    .reportview-container {
        background-color: #f5f5f5;
        color: #333;
    }
    .main .block-container {
        max-width: 1200px;
        padding: 20px;
    }
    header {
        background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
        color: white;
        padding: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        border-radius: 8px; /* Added for a softer look */
    }
    h1 {
        margin: 0;
        font-size: 2.5rem;
        text-align: center;
        color: white !important; /* Ensure heading is white */
    }
    .subtitle {
        text-align: center;
        opacity: 0.9;
        margin-top: 10px;
        color: white;
    }
    .stCard { /* Streamlit's equivalent for custom cards */
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        font-size: 16px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    textarea { /* Targeting Streamlit's textarea */
        width: 100%;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 16px;
        min-height: 150px;
        margin-bottom: 20px;
    }
    .result-fake {
        background-color: #ffebee;
        color: #c62828;
        border-left: 5px solid #c62828;
        padding: 15px;
        border-radius: 4px;
        font-weight: bold;
        margin-top: 20px;
    }
    .result-real {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 4px;
        font-weight: bold;
        margin-top: 20px;
    }
    .feature-card {
        flex: 1;
        min-width: 250px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        text-align: center;
        margin-bottom: 20px; /* Added margin for spacing */
    }
    .feature-card i {
        font-size: 40px;
        color: #9d50bb;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #666;
        font-size: 14px;
    }
    /* Streamlit specific adjustments */
    div.stButton {
        text-align: center;
    }
    .stMarkdown h2 {
        font-size: 1.8rem; /* Adjust heading size */
        color: #333;
    }
    .css-1jc7ptx.e1ewe7hr3 { /* Target specific Streamlit elements for padding */
        padding-top: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<header>
    <h1>COVID-19 Hoax Detection</h1>
    <p class="subtitle">Deteksi berita palsu tentang COVID-19 menggunakan model NLP berbasis SGD</p>
</header>
""", unsafe_allow_html=True)

# --- Analisis Teks Section ---
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.markdown('<h2><i class="fas fa-search"></i> Analisis Teks</h2>', unsafe_allow_html=True)
st.write("Masukkan teks berita atau headline yang ingin diperiksa kebenarannya:")

input_text = st.text_area(" ", placeholder="Masukkan teks berita disini...", height=150)

if st.button("Analisis Sekarang"):
    if input_text.strip() == '':
        st.warning('Silakan masukkan teks yang ingin dianalisis!')
    else:
        # Simulate AI processing (replace with actual model inference in a real application)
        is_fake = random.random() > 0.5
        confidence = round(random.uniform(80, 100), 2)

        st.markdown('<div id="result" class="result-{}">'.format("fake" if is_fake else "real"), unsafe_allow_html=True)
        st.markdown("<h3>Hasil Analisis:</h3>", unsafe_allow_html=True)
        if is_fake:
            st.markdown('<p id="resultText"><i class="fas fa-times-circle"></i> BERITA INI DIIDENTIFIKASI SEBAGAI BERITA PALSU</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p id="resultText"><i class="fas fa-check-circle"></i> BERITA INI DIIDENTIFIKASI SEBAGAI BERITA ASLI</p>', unsafe_allow_html=True)
        st.markdown(f'<p id="confidence">Tingkat Kepercayaan: {confidence}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Visualisasi Data Section ---
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.markdown('<h2><i class="fas fa-chart-pie"></i> Visualisasi Data</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Distribusi Label", "Word Cloud", "Panjang Teks"])

with tab1:
    st.markdown('<div class="visualization">', unsafe_allow_html=True)
    st.markdown("<h3>Distribusi Berita Palsu vs Asli</h3>", unsafe_allow_html=True)
    st.image("https://via.placeholder.com/800x400?text=Grafik+Distribusi+Label", caption="Distribusi Label", use_column_width=True)
    st.write("Grafik menunjukkan ketidakseimbangan jumlah data antara berita asli dan palsu dalam dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="visualization">', unsafe_allow_html=True)
    st.markdown("<h3>Word Cloud Berita Palsu</h3>", unsafe_allow_html=True)
    st.image("https://via.placeholder.com/800x400?text=Word+Cloud+Hoax", caption="Word Cloud Hoax", use_column_width=True)
    st.markdown("<h3>Word Cloud Berita Asli</h3>", unsafe_allow_html=True)
    st.image("https://via.placeholder.com/800x400?text=Word+Cloud+Real", caption="Word Cloud Real", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="visualization">', unsafe_allow_html=True)
    st.markdown("<h3>Distribusi Panjang Teks</h3>", unsafe_allow_html=True)
    st.image("https://via.placeholder.com/800x400?text=Histogram+Panjang+Teks", caption="Panjang Teks", use_column_width=True)
    st.write("Histogram menunjukkan distribusi panjang teks (jumlah karakter) dari berita dalam dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close stCard for Visualization

# --- Features Section ---
st.markdown('<div class="features">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <i class="fas fa-brain"></i>
        <h3>Model SGD</h3>
        <p>Menggunakan Stochastic Gradient Descent yang dioptimalkan untuk klasifikasi teks dengan akurasi tinggi (>80%).</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <i class="fas fa-language"></i>
        <h3>NLP Pipeline</h3>
        <p>Preprocessing teks dengan tokenisasi, stopword removal, dan TF-IDF untuk representasi numerik.</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <i class="fas fa-chart-bar"></i>
        <h3>Analisis Mendalam</h3>
        <p>Menyediakan visualisasi data dan metrik evaluasi untuk memahami performa model.</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True) # Close features div

# --- NLP Pipeline Section ---
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.markdown('<h2><i class="fas fa-project-diagram"></i> NLP Pipeline</h2>', unsafe_allow_html=True)
st.image("https://via.placeholder.com/800x300?text=NLP+Pipeline+Diagram", caption="NLP Pipeline", use_column_width=True)
st.markdown("""
<ol>
    <li><strong>Preprocessing:</strong> Membersihkan teks dari karakter tidak penting, mengubah ke huruf kecil, dan tokenisasi.</li>
    <li><strong>Feature Extraction:</strong> Mengubah teks menjadi representasi numerik menggunakan TF-IDF.</li>
    <li><strong>Model Training:</strong> Melatih model SGD dengan data yang sudah diproses.</li>
    <li><strong>Evaluation:</strong> Mengevaluasi model menggunakan metrik akurasi, presisi, recall, dan F1-score.</li>
    <li><strong>Deployment:</strong> Menerapkan model kedalam aplikasi untuk klasifikasi real-time.</li>
</ol>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("<p>Aplikasi ini dikembangkan oleh Azka Mayla Fayza (A11.2022.14298) menggunakan dataset dari Kaggle</p>", unsafe_allow_html=True)
st.markdown('<p><a href="https://www.kaggle.com/code/moathmohamed/covid-19-fake-news-detection-96/input" target="_blank">Link Dataset</a></p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Font Awesome CDN ---
# Streamlit doesn't directly support <link> in markdown for external CSS,
# but Font Awesome is often added in the head for full control.
# For simplicity, we can sometimes inject it via markdown if the content security policy allows,
# or more robustly, use st.components.v1.html for custom head content if necessary.
# For now, relying on the original CSS link to be loaded by Streamlit's rendering
# or assuming Font Awesome is pre-loaded in the environment.
# Since it's a direct HTML conversion, the best practice is to put it in a way
# Streamlit can render it, which is often done via st.markdown with unsafe_allow_html=True.
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)
