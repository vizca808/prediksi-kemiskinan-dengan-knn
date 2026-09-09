# Prediksi Tingkat Kemiskinan di Indonesia Menggunakan Algoritma K-Nearest Neighbors (KNN)

Proyek ini memprediksi tingkat kemiskinan per kabupaten/kota di Indonesia berdasarkan data indikator sosial ekonomi (IPM, Rata-rata Lama Sekolah, Pengeluaran per Kapita, dsb.) menggunakan algoritma **K-Nearest Neighbors (KNN)**.

Notebook asli diambil dari Kaggle:
- **Notebook**: [Prediksi Tingkat kemiskinan Dengan Knn](https://www.kaggle.com/code/faisaldinobahtiar/prediksi-tingkat-kemiskinan-dengan-knn)
- **Author**: Faisal Dino Bahtiar
- **Dataset**: [Klasifikasi Tingkat Kemiskinan di Indonesia](https://www.kaggle.com/datasets/ermila/klasifikasi-tingkat-kemiskinan-di-indonesia)

---

## Struktur Direktori

\\	ext
├── data/
│   └── Klasifikasi Tingkat Kemiskinan di Indonesia.csv
├── prediksi-tingkat-kemiskinan-dengan-knn.ipynb
├── main.py
├── requirements.txt
└── README.md
\
---

## Alur Kerja Proyek

1. **Eksplorasi & Pembersihan Data (EDA)**:
   - Mengubah tanda koma desimal menjadi titik dan parsing numerik.
   - Analisis korelasi fitur & heatmap.
   - Analisis persebaran kelas target (\Klasifikasi_Kemiskinan\).
2. **Preprocessing**:
   - One-Hot Encoding fitur kategorikal.
   - Normalisasi fitur numerik menggunakan \MinMaxScaler\.
   - Reduksi dimensi menggunakan PCA (_components=10\).
3. **Modeling & Validasi**:
   - Train-Test Split (80:20 stratifikasi).
   - Pencarian nilai k optimal melalui Cross-Validation (ange(1, 21)\).
   - Training \KNeighborsClassifier\ dengan k terbaik.
4. **Evaluasi**:
   - Akurasi, Presisi, Recall, dan Confusion Matrix.

---

## Cara Menjalankan

### 1. Install Dependensi
\\ash
pip install -r requirements.txt
\
### 2. Jalankan Notebook / Script
- **Jupyter Notebook**: Buka \prediksi-tingkat-kemiskinan-dengan-knn.ipynb- **Python Script**:
  \\ash
  python main.py
  \