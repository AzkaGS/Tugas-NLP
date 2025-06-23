# -*- coding: utf-8 -*-
"""
Klasifikasi Hoaks COVID-19 dengan Model SGD
Studi Kasus: Klasifikasi hoaks COVID-19 terbaru dengan model SGD
"""

# ===== INSTALASI DEPENDENSI =====
# Menginstal library yang diperlukan. Pastikan semua dependensi terinstal.
# scikit-learn: Untuk algoritma machine learning dan utilitas.
# pandas: Untuk manipulasi dan analisis data.
# numpy: Untuk komputasi numerik.
# matplotlib: Untuk visualisasi data.
# seaborn: Untuk visualisasi statistik yang lebih indah.
# wordcloud: Untuk membuat word cloud dari teks.
# imblearn: Untuk menangani ketidakseimbangan kelas (SMOTE).
!pip install scikit-learn pandas numpy matplotlib seaborn wordcloud imblearn

# ===== IMPOR LIBRARY =====
# Mengimpor semua library yang akan digunakan dalam proyek ini.
import pandas as pd # Untuk DataFrame dan manipulasi data
import numpy as np # Untuk operasi array numerik
import matplotlib.pyplot as plt # Untuk membuat plot dan grafik
import seaborn as sns # Untuk visualisasi data statistik tingkat tinggi
from wordcloud import WordCloud # Untuk menghasilkan word clouds
import warnings # Untuk mengelola peringatan
warnings.filterwarnings('ignore') # Mengabaikan peringatan untuk output yang lebih bersih

# Pemrosesan Teks (NLP)
import re # Untuk operasi ekspresi reguler
import string # Untuk string konstanta (misalnya, tanda baca)
from nltk.corpus import stopwords # Untuk daftar kata-kata yang umum (stop words)
from nltk.tokenize import word_tokenize # Untuk memecah teks menjadi token (kata-kata)
from nltk.stem import PorterStemmer # Untuk stemming kata (mengurangi kata ke akar)
import nltk # Toolkit Natural Language

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer # Untuk mengonversi teks menjadi fitur numerik (TF-IDF)
from sklearn.model_selection import train_test_split, cross_val_score # Untuk membagi data dan validasi silang
from sklearn.linear_model import SGDClassifier, LogisticRegression # Algoritma Klasifikasi SGD dan Regresi Logistik
from sklearn.svm import SVC # Algoritma Support Vector Machine
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Metrik evaluasi model
from sklearn.pipeline import Pipeline # Untuk membuat alur kerja ML
from sklearn.preprocessing import LabelEncoder # Untuk mengkodekan label kategorikal menjadi numerik
from imblearn.over_sampling import SMOTE # Untuk oversampling data minoritas (menangani ketidakseimbangan kelas)
import joblib # Untuk menyimpan dan memuat model (misalnya, TF-IDF dan model terlatih)

# Mengunduh data NLTK yang diperlukan. Ini penting untuk pemrosesan teks.
nltk.download('punkt') # Untuk tokenisasi
nltk.download('stopwords') # Untuk daftar stop words
nltk.download('wordnet') # Untuk lemmatisasi (meskipun PorterStemmer digunakan di sini, WordNet berguna untuk lemmatisasi yang lebih baik)
nltk.download('punkt_tab') # Data tambahan untuk tokenisasi

# ===== MUAT DATASET =====
# Bagian ini mencoba memuat dataset dari file CSV.
# Jika file tidak ditemukan, ia akan membuat contoh dataset untuk demonstrasi.
# Pastikan Anda telah mengunggah file 'data.csv' ke lingkungan Anda (misalnya Google Colab).

try:
    # Memuat dataset dari path yang ditentukan
    df = pd.read_csv('/content/data.csv')
    print("Dataset berhasil dimuat!")
except FileNotFoundError:
    # Jika file tidak ditemukan, buat dataset sampel
    print("File tidak ditemukan. Silakan upload file 'data.csv' terlebih dahulu.")
    sample_data = {
        'headlines': [
            'COVID-19 vaccine causes autism and brain damage',
            'New COVID-19 variant more dangerous than previous ones',
            'Drinking hot water can cure coronavirus infection',
            'WHO announces new COVID-19 prevention guidelines',
            'Garlic and ginger can completely prevent COVID-19',
            'Hospitals report increase in COVID-19 cases',
            'Sunlight kills coronavirus in minutes, scientists say',
            'Health ministry updates COVID-19 vaccination schedule',
            'Miracle cure for COVID-19 discovered in traditional medicine',
            'Research shows effectiveness of COVID-19 vaccines',
        ],
        'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 0: Hoaks, 1: Nyata
    }
    df = pd.DataFrame(sample_data)
    print("Menggunakan sample dataset untuk demonstrasi.")

# Menampilkan bentuk dataset (jumlah baris, jumlah kolom)
print(f"Dataset shape: {df.shape}")
# Menampilkan nama-nama kolom dalam dataset
print(f"Columns: {df.columns.tolist()}")

# Memastikan kolom 'outcome' bertipe integer
df['outcome'] = df['outcome'].astype(int)


# ===== ANALISIS DATA EKSPLORATIF (EDA) =====
# Bagian ini melakukan analisis awal pada data untuk memahami strukturnya,
# mendeteksi nilai yang hilang, dan melihat distribusi data.
print("\n===== EXPLORATORY DATA ANALYSIS =====")

# 1. Informasi dasar tentang dataset
print("Dataset Info:")
print(df.info()) # Menampilkan tipe data, entri non-null, dan penggunaan memori
print("\nDataset Description:")
print(df.describe()) # Menampilkan statistik deskriptif untuk kolom numerik

# 2. Periksa nilai yang hilang
print(f"\nMissing values:")
print(df.isnull().sum()) # Menghitung jumlah nilai null di setiap kolom

# 3. Periksa distribusi variabel target ('outcome')
print(f"\nTarget distribution:")
print(df['outcome'].value_counts()) # Menghitung berapa kali setiap nilai unik muncul di kolom 'outcome'

# 4. Contoh data teratas
print(f"\nSample data:")
print(df.head()) # Menampilkan 5 baris pertama dataset

# ===== VISUALISASI DATA =====
# Bagian ini membuat beberapa plot untuk memvisualisasikan aspek-aspek penting dari data.
print("\n===== DATA VISUALIZATION =====")

# Mengatur gaya plot untuk estetika yang lebih baik
plt.style.use('seaborn-v0_8')
# Membuat subplot 2x2 untuk menampung beberapa plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribusi label (Hoaks vs Nyata)
# Membuat bar plot dari distribusi nilai 'outcome'
df['outcome'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
axes[0,0].set_title('Distribusi Label (Hoaks vs Nyata)') # Judul plot
axes[0,0].set_ylabel('Jumlah') # Label sumbu Y
axes[0,0].set_xlabel('Label') # Label sumbu X
axes[0,0].tick_params(axis='x', rotation=45) # Memutar label sumbu X

# 2. Distribusi panjang teks
# Menghitung panjang setiap teks di kolom 'headlines'
df['text_length'] = df['headlines'].str.len()
# Membuat histogram distribusi panjang teks
df['text_length'].hist(bins=30, ax=axes[0,1], color='lightgreen', alpha=0.7)
axes[0,1].set_title('Distribusi Panjang Teks')
axes[0,1].set_ylabel('Frekuensi')
axes[0,1].set_xlabel('Panjang Teks')

# 3. Heatmap nilai yang hilang
# Membuat heatmap untuk memvisualisasikan nilai-nilai yang hilang dalam dataset
sns.heatmap(df.isnull(), cbar=True, ax=axes[1,0], cmap='viridis')
axes[1,0].set_title('Missing Values Heatmap')

# 4. Panjang teks berdasarkan label
# Membuat box plot untuk membandingkan panjang teks antara berita hoaks dan berita nyata
df.boxplot(column='text_length', by='outcome', ax=axes[1,1])
axes[1,1].set_title('Panjang Teks berdasarkan Label')
axes[1,1].set_ylabel('Panjang Teks')

plt.tight_layout() # Menyesuaikan tata letak plot agar tidak tumpang tindih
plt.show() # Menampilkan plot

# ===== PRA-PEMROSESAN TEKS =====
# Bagian ini mendefinisikan dan menerapkan langkah-langkah pra-pemrosesan pada teks.
# Tujuannya adalah untuk membersihkan teks dan mengubahnya menjadi format yang lebih cocok untuk model ML.
print("\n===== TEXT PREPROCESSING =====")

class TextPreprocessor:
    """
    Kelas untuk membersihkan dan memproses teks.
    Meliputi konversi ke huruf kecil, penghapusan URL, tanda baca, angka,
    tokenisasi, penghapusan stop words, dan stemming.
    """
    def __init__(self):
        # Menginisialisasi stemmer Porter
        self.stemmer = PorterStemmer()
        # Mengatur stop words bahasa Inggris dari NLTK
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Membersihkan teks dari karakter yang tidak diinginkan seperti URL,
        mention, hashtag, angka, dan tanda baca. Mengubah teks menjadi huruf kecil
        dan menghapus spasi ekstra.
        """
        if pd.isna(text): # Menangani nilai NaN (Not a Number)
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

    def tokenize_and_stem(self, text):
        """
        Melakukan tokenisasi teks (memecah menjadi kata-kata) dan kemudian
        melakukan stemming pada setiap token. Juga menghapus stop words.
        """
        # Melakukan tokenisasi teks
        tokens = word_tokenize(text)
        # Melakukan stemming pada setiap token yang bukan stop word
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        # Menggabungkan kembali token yang sudah diproses menjadi sebuah string
        return ' '.join(tokens)

    def preprocess(self, text):
        """
        Pipa pra-pemrosesan lengkap yang menggabungkan langkah-langkah pembersihan,
        tokenisasi, dan stemming.
        """
        text = self.clean_text(text) # Langkah pembersihan
        text = self.tokenize_and_stem(text) # Langkah tokenisasi dan stemming
        return text

# Menginisialisasi objek preprocessor
preprocessor = TextPreprocessor()

# Menerapkan pra-pemrosesan pada kolom 'headlines' dan menyimpan hasilnya di 'processed_text'
print("Memproses data teks...")
df['processed_text'] = df['headlines'].apply(preprocessor.preprocess)

# Menampilkan beberapa contoh hasil pra-pemrosesan
print("\nContoh preprocessing:")
for i in range(3):
    print(f"Original: {df['headlines'].iloc[i]}")
    print(f"Processed: {df['processed_text'].iloc[i]}")
    print("-" * 50)

# ===== REKAYASA FITUR =====
# Bagian ini menambahkan fitur-fitur baru ke dataset berdasarkan teks yang ada.
# Fitur-fitur ini dapat memberikan informasi tambahan untuk model.
print("\n===== FEATURE ENGINEERING =====")

# Menambahkan fitur 'word_count' (jumlah kata)
df['word_count'] = df['headlines'].str.split().str.len()
# Menambahkan fitur 'char_count' (jumlah karakter)
df['char_count'] = df['headlines'].str.len()
# Menambahkan fitur 'avg_word_length' (rata-rata panjang kata)
df['avg_word_length'] = df['char_count'] / df['word_count']
# Mengisi nilai NaN yang mungkin muncul (misalnya jika word_count adalah 0) dengan 0
df['avg_word_length'] = df['avg_word_length'].fillna(0)

print("Fitur tambahan dibuat:")
print(f"Rentang jumlah kata: {df['word_count'].min()} - {df['word_count'].max()}")
print(f"Rentang jumlah karakter: {df['char_count'].min()} - {df['char_count'].max()}")

# ===== SIAPKAN DATA UNTUK PEMODELAN =====
# Bagian ini menyiapkan data menjadi format yang dibutuhkan oleh model machine learning.
# Ini melibatkan pengkodean label dan pembagian data menjadi set pelatihan dan pengujian.
print("\n===== PREPARE DATA FOR MODELING =====")

# Mengkodekan variabel target 'outcome' (0 atau 1) menjadi 'target' menggunakan LabelEncoder
# Ini memastikan bahwa target adalah numerik dan berurutan.
le = LabelEncoder()
df['target'] = le.fit_transform(df['outcome'])

# Mendefinisikan fitur (X) dan target (y)
X = df['processed_text'] # Fitur adalah teks yang telah diproses
y = df['target'] # Target adalah label yang sudah dienkode

# Membagi data menjadi set pelatihan (80%) dan set pengujian (20%)
# random_state: Untuk reproduktifitas hasil (pembagian yang sama setiap kali dijalankan)
# stratify: Memastikan distribusi kelas yang sama di set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Ukuran set pelatihan: {len(X_train)}")
print(f"Ukuran set pengujian: {len(X_test)}")

# ===== PEMBANGUNAN MODEL =====
# Bagian ini menyiapkan vektorizer TF-IDF dan menangani ketidakseimbangan kelas.
print("\n===== MODEL BUILDING =====")

# Menginisialisasi TfidfVectorizer
# max_features: Membatasi jumlah fitur (kata/ngram) yang akan digunakan.
# min_df: Mengabaikan istilah yang frekuensinya kurang dari ambang batas ini.
# max_df: Mengabaikan istilah yang frekuensinya lebih dari ambang batas ini.
# ngram_range: Menggunakan unigram (kata tunggal) dan bigram (pasangan kata).
# stop_words: Menghapus stop words bahasa Inggris selama vektorisasi.
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

# Mengubah data teks pelatihan menjadi matriks TF-IDF
X_train_tfidf = tfidf.fit_transform(X_train)
# Mengubah data teks pengujian menggunakan vektorizer yang sudah dilatih (TRANSFORM saja, bukan FIT_TRANSFORM)
X_test_tfidf = tfidf.transform(X_test)

print(f"Bentuk matriks TF-IDF (pelatihan): {X_train_tfidf.shape}")

# Menangani ketidakseimbangan kelas menggunakan SMOTE (Synthetic Minority Over-sampling Technique)
# SMOTE menghasilkan sampel baru dari kelas minoritas untuk menyeimbangkan distribusi kelas.
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

print(f"Distribusi set pelatihan asli: {np.bincount(y_train)}")
print(f"Distribusi set pelatihan yang seimbang (setelah SMOTE): {np.bincount(y_train_balanced)}")

# ===== PELATIHAN MODEL =====
# Bagian ini mendefinisikan, melatih, dan mengevaluasi model klasifikasi yang dipilih.
print("\n===== MODEL TRAINING =====")

# Mendefinisikan model-model yang akan digunakan: SGDClassifier, LogisticRegression, SVC (SVM).
models = {
    'SGD': SGDClassifier(loss='hinge', alpha=0.001, random_state=42, max_iter=1000),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(kernel='linear', random_state=42, probability=True) # probability=True diperlukan untuk predict_proba
}

# Kamus untuk menyimpan hasil dan prediksi dari setiap model
results = {}
predictions = {}

# Melatih dan mengevaluasi setiap model dalam kamus 'models'
for name, model in models.items():
    print(f"\nMelatih model: {name}...")

    # Melatih model menggunakan data pelatihan yang sudah diseimbangkan oleh SMOTE
    model.fit(X_train_balanced, y_train_balanced)

    # Membuat prediksi pada set pengujian
    y_pred = model.predict(X_test_tfidf)
    # Mendapatkan probabilitas prediksi jika model mendukungnya
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None

    # Menghitung akurasi model
    accuracy = accuracy_score(y_test, y_pred)

    # Melakukan validasi silang (cross-validation) untuk evaluasi yang lebih robust
    # cv=5 berarti data dibagi menjadi 5 fold, dan model dilatih/diuji 5 kali
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)

    # Menyimpan hasil evaluasi dan prediksi
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    predictions[name] = y_pred

    print(f"{name} Akurasi: {accuracy:.4f}")
    print(f"{name} Skor Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Setelah semua model dilatih, tentukan model terbaik berdasarkan akurasi
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
# Simpan model terbaik dan TF-IDF vectorizer untuk deployment
best_model = models[best_model_name]

# Menyimpan model terbaik dan TF-IDF vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(preprocessor, 'text_preprocessor.pkl') # Simpan juga preprocessor
joblib.dump(le, 'label_encoder.pkl') # Simpan juga label encoder

print(f"\nModel terbaik ({best_model_name}), TF-IDF Vectorizer, Text Preprocessor, dan Label Encoder telah disimpan.")


# ===== EVALUASI MODEL =====
# Bagian ini membuat visualisasi untuk membandingkan kinerja model dan menganalisis hasil.
print("\n===== MODEL EVALUATION =====")

# Membuat subplot 2x2 untuk visualisasi evaluasi
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Perbandingan Akurasi Model
model_names = list(results.keys()) # Nama-nama model
accuracies = [results[name]['accuracy'] for name in model_names] # Akurasi masing-masing model

axes[0,0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral']) # Bar plot akurasi
axes[0,0].set_title('Perbandingan Akurasi Model')
axes[0,0].set_ylabel('Akurasi')
axes[0,0].set_ylim(0, 1) # Batas sumbu Y dari 0 sampai 1
for i, v in enumerate(accuracies):
    axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom') # Menampilkan nilai akurasi di atas bar

# 2. Confusion Matrix untuk Model Terbaik
# Memilih model dengan akurasi tertinggi sebagai 'best_model'
best_model_name_for_cm = max(results, key=lambda x: results[x]['accuracy'])
cm = confusion_matrix(y_test, results[best_model_name_for_cm]['predictions']) # Menghitung confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1]) # Heatmap confusion matrix
axes[0,1].set_title(f'Confusion Matrix - {best_model_name_for_cm}')
axes[0,1].set_ylabel('True Label')
axes[0,1].set_xlabel('Predicted Label')

# 3. Skor Cross-Validation
cv_means = [results[name]['cv_mean'] for name in model_names] # Rata-rata skor CV
cv_stds = [results[name]['cv_std'] for name in model_names] # Standar deviasi skor CV

axes[1,0].bar(model_names, cv_means, yerr=cv_stds, capsize=5,
              color=['skyblue', 'lightgreen', 'lightcoral']) # Bar plot skor CV dengan error bars
axes[1,0].set_title('Cross-Validation Scores')
axes[1,0].set_ylabel('CV Score')
axes[1,0].set_ylim(0, 1)

# 4. Plot ini sengaja dikosongkan atau dapat digunakan untuk plot lain jika diperlukan.
# Contoh: Jika kita ingin menampilkan fitur penting dari model tertentu (misalnya, jika Random Forest ada)
# Untuk saat ini, kita biarkan kosong atau berikan placeholder
axes[1,1].set_title('Plot Tambahan (Saat ini kosong)')
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])

plt.tight_layout() # Menyesuaikan tata letak plot
plt.show() # Menampilkan plot

# ===== LAPORAN KLASIFIKASI RINCI =====
# Bagian ini mencetak laporan klasifikasi lengkap (presisi, recall, f1-score)
# untuk setiap model yang dilatih.
print("\n===== DETAILED CLASSIFICATION REPORTS =====")

for name, model in models.items():
    print(f"\n{name} Laporan Klasifikasi:")
    print("-" * 50)
    # Menghasilkan laporan klasifikasi
    report = classification_report(y_test, results[name]['predictions'],
                                   target_names=['hoaks', 'nyata']) # Menggunakan label string untuk kejelasan
    print(report)

# ===== WORD CLOUDS =====
# Bagian ini menghasilkan word cloud untuk memvisualisasikan kata-kata yang paling sering muncul
# dalam berita hoaks dan berita nyata.
print("\n===== WORD CLOUDS =====")

# Menggabungkan semua teks yang diproses untuk berita hoaks (outcome == 0)
fake_text = ' '.join(df[df['outcome'] == 0]['processed_text'].astype(str))
# Menggabungkan semua teks yang diproses untuk berita nyata (outcome == 1)
real_text = ' '.join(df[df['outcome'] == 1]['processed_text'].astype(str))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Word Cloud untuk Berita Hoaks
if fake_text.strip(): # Memastikan ada teks yang valid
    wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
    axes[0].imshow(wordcloud_fake, interpolation='bilinear')
    axes[0].set_title('Word Cloud - Berita Hoaks')
    axes[0].axis('off') # Menyembunyikan sumbu

# Word Cloud untuk Berita Nyata
if real_text.strip(): # Memastikan ada teks yang valid
    wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
    axes[1].imshow(wordcloud_real, interpolation='bilinear')
    axes[1].set_title('Word Cloud - Berita Nyata')
    axes[1].axis('off') # Menyembunyikan sumbu

plt.tight_layout()
plt.show()

# ===== ANALISIS KESALAHAN =====
# Bagian ini menganalisis contoh-contoh di mana model membuat prediksi yang salah.
print("\n===== ERROR ANALYSIS =====")

# Mengambil nama model terbaik yang ditentukan sebelumnya
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_predictions = results[best_model_name]['predictions']

# Menemukan indeks di mana prediksi model terbaik tidak cocok dengan label sebenarnya
misclassified_idx = np.where(y_test != best_predictions)[0]

print(f"Jumlah kesalahan klasifikasi: {len(misclassified_idx)}")
print(f"Persentase kesalahan: {len(misclassified_idx)/len(y_test)*100:.2f}%")

if len(misclassified_idx) > 0:
    print("\nContoh kesalahan klasifikasi:")
    # Menampilkan 5 contoh kesalahan pertama
    # Mengambil indeks asli dari DataFrame untuk contoh-contoh yang salah diklasifikasikan
    test_indices = X_test.iloc[misclassified_idx[:5]].index

    for idx in test_indices:
        # Mengambil label sebenarnya dari DataFrame asli
        true_label = le.inverse_transform([df.loc[idx, 'target']])[0]
        # Mengambil prediksi model (perlu menemukan indeks prediksi yang sesuai dengan indeks asli df)
        # y_test adalah Series dengan indeks asli dari df, jadi kita bisa mencocokkan indeks
        pred_label_index_in_y_test = (y_test.index == idx).argmax()
        pred_label = le.inverse_transform([best_predictions[pred_label_index_in_y_test]])[0]


        print(f"\nOriginal Text: {df.loc[idx, 'headlines']}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {pred_label}")
        print("-" * 80)

# ===== RINGKASAN DAN REKOMENDASI =====
# Bagian ini memberikan ringkasan hasil proyek dan rekomendasi untuk peningkatan di masa mendatang.
print("\n===== SUMMARY AND RECOMMENDATIONS =====")

print("KESIMPULAN:")
print("=" * 50)
print(f"Model terbaik yang dipilih berdasarkan akurasi pada set pengujian: {best_model_name}")
print(f"Akurasi model terbaik: {results[best_model_name]['accuracy']:.4f}")
print(f"Rata-rata Skor Cross-Validation (5-fold) untuk model terbaik: {results[best_model_name]['cv_mean']:.4f}")

print("\nPeringkat Model berdasarkan Akurasi (tertinggi ke terendah):")
# Mengurutkan model berdasarkan akurasi secara menurun
sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (name, result) in enumerate(sorted_models, 1):
    print(f"{i}. {name}: Akurasi = {result['accuracy']:.4f}, CV Mean = {result['cv_mean']:.4f}")

print("\nREKOMENDASI UNTUK PENINGKATAN MASA DEPAN:")
print("=" * 50)
print("1. Ensembling Model: Gabungkan prediksi dari beberapa model terbaik (misalnya, voting classifier) untuk meningkatkan robustnes dan akurasi.")
print("2. Augmentasi Data: Jika dataset kecil, pertimbangkan teknik augmentasi teks (misalnya, sinonim, back-translation) untuk memperbanyak data pelatihan.")
print("3. Penyetelan Hyperparameter: Lakukan tuning hyperparameter yang lebih ekstensif menggunakan GridSearchCV atau RandomizedSearchCV untuk setiap model, khususnya untuk SGDClassifier (alpha, penalty) dan SVC (C, gamma).")
print("4. Model Deep Learning: Eksplorasi model berbasis Transformer seperti BERT atau RoBERTa. Meskipun membutuhkan lebih banyak sumber daya, mereka seringkali memberikan kinerja yang superior untuk tugas NLP.")
print("5. Fitur Tambahan: Sertakan fitur tambahan seperti POS tagging, named entity recognition, atau fitur gaya penulisan untuk memperkaya representasi teks.")
print("6. Analisis Lebih Lanjut tentang Misklasifikasi: Selidiki lebih dalam contoh-contoh yang salah diklasifikasikan untuk mengidentifikasi pola atau jenis kesalahan yang umum dan memperbaiki model sesuai kebutuhan.")
print("7. Pembaruan Stop Words: Sesuaikan daftar stop words atau buat daftar stop words kustom yang lebih relevan dengan konteks COVID-19.")
print("8. Deteksi Outlier: Identifikasi dan tangani outlier dalam data teks yang mungkin memengaruhi kinerja model.")

# ===== SIMPAN HASIL =====
# Bagian ini menyimpan hasil evaluasi model dan prediksi ke file CSV.
print("\n===== SAVE RESULTS =====")

# Membuat DataFrame dari hasil model
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [results[name]['accuracy'] for name in model_names],
    'CV_Mean': [results[name]['cv_mean'] for name in model_names],
    'CV_Std': [results[name]['cv_std'] for name in model_names]
})

# Menyimpan hasil ke file CSV
results_df.to_csv('model_results.csv', index=False)
print("Hasil evaluasi model disimpan ke 'model_results.csv'")

# Membuat DataFrame dari prediksi
predictions_df = pd.DataFrame(predictions)
# Menambahkan kolom label sebenarnya ke DataFrame prediksi
predictions_df['True_Label'] = y_test.values # Menggunakan .values untuk memastikan keselarasan indeks
predictions_df.to_csv('predictions.csv', index=False)
print("Prediksi model disimpan ke 'predictions.csv'")

print("\n===== PROYEK SELESAI =====")
print("Proyek Klasifikasi Hoaks COVID-19 telah selesai!")
print("Semua visualisasi, analisis, dan hasil telah ditampilkan dan disimpan.")
print("Sekarang Anda dapat menggunakan model yang disimpan untuk deployment.")
