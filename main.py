# =============================================================================
# Prediksi Tingkat Kemiskinan di Indonesia Menggunakan Algoritma K-Nearest Neighbors (KNN)
# Original Author: Faisal Dino Bahtiar (Kaggle)
# =============================================================================


# ----------------------------------------------------------------------
# **Import Library Dan Dataset**
# ----------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


import os
data_path = 'data/Klasifikasi Tingkat Kemiskinan di Indonesia.csv'
if not os.path.exists(data_path):
    data_path = '/kaggle/input/klasifikasi-tingkat-kemiskinan-di-indonesia/Klasifikasi Tingkat Kemiskinan di Indonesia.csv'
df = pd.read_csv(data_path, delimiter=';')
df.head()



# ----------------------------------------------------------------------
# **EDA(Exploratory Data Analysis)**
# ----------------------------------------------------------------------

# Data Cleaning: Replace commas with dots and convert to numeric
cols_to_convert = df.columns[2:]
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)


# Rename columns for simplicity
df.columns = [
    'Provinsi', 'Kab/Kota', 'Persentase_Miskin', 'Rata2_Lama_Sekolah', 
    'Pengeluaran_per_Kapita', 'IPM', 'Umur_Harapan_Hidup', 'Akses_Sanitasi', 
    'Akses_Air_Minum', 'Tingkat_Pengangguran', 'TPAK', 'PDRB', 'Klasifikasi_Kemiskinan'
]


# Drop missing values
df_cleaned = df.dropna()


# Data Exploration: Heatmap Correlation
plt.figure(figsize=(12, 8))
sns.heatmap(df_cleaned.drop(columns=['Provinsi', 'Kab/Kota']).corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap Korelasi Antar Variabel")
plt.show()



# ----------------------------------------------------------------------
# Dari heatmap korelasi, beberapa hal yang menarik:
# 
# * Persentase Kemiskinan berkorelasi negatif dengan IPM (-0.74), Pengeluaran per Kapita (-0.67), dan Akses Sanitasi (-0.61), menunjukkan bahwa semakin tinggi nilai-nilai tersebut, semakin rendah tingkat kemiskinan.
# * Pengeluaran per Kapita dan PDRB memiliki korelasi tinggi (0.78), wajar karena PDRB mencerminkan kondisi ekonomi daerah.
# * Rata-rata Lama Sekolah berkorelasi positif dengan IPM (0.85), menegaskan bahwa pendidikan adalah faktor utama dalam pembangunan manusia.
# ----------------------------------------------------------------------

# Visualisasi distribusi kelas target
plt.figure(figsize=(6, 4))
sns.countplot(x=df_cleaned["Klasifikasi_Kemiskinan"], palette="pastel")
plt.xticks([0, 1], ["Tidak Miskin", "Miskin"])
plt.title("Distribusi Kelas Klasifikasi Kemiskinan")
plt.xlabel("Kategori")
plt.ylabel("Jumlah")
plt.show()



# ----------------------------------------------------------------------
# Distribusi kelas target terlihat cukup seimbang antara kategori "Tidak Miskin" (0) dan "Miskin" (1), sehingga model KNN nantinya tidak akan terlalu bias ke salah satu kelas.
# ----------------------------------------------------------------------

# Encoding categorical variables (One-Hot Encoding)
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_categorical = encoder.fit_transform(df_cleaned[['Provinsi', 'Kab/Kota']])
df_encoded = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(['Provinsi', 'Kab/Kota']), index=df_cleaned.index)


df_final = pd.concat([df_cleaned.drop(columns=['Provinsi', 'Kab/Kota']), df_encoded], axis=1)



# ----------------------------------------------------------------------
# **Training Dan Modeling Data**
# ----------------------------------------------------------------------

# Normalization
X = df_final.drop(columns=['Klasifikasi_Kemiskinan'])
y = df_final['Klasifikasi_Kemiskinan']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


# PCA for Dimensionality Reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled_df)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)


# Finding the Best K using Cross Validation
k_values = range(1, 21)
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5, scoring='accuracy').mean() for k in k_values]
best_k = k_values[np.argmax(cv_scores)]



# ----------------------------------------------------------------------
# Nilai K terbaik adalah 10, berdasarkan cross-validation.
# ----------------------------------------------------------------------

# Train KNN Model
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)



# ----------------------------------------------------------------------
# **Hasil Evaluasi**
# ----------------------------------------------------------------------

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


print(f"Best K: {best_k}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
print(f"ROC AUC: {roc_auc:.2%}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)



# ----------------------------------------------------------------------
# Hasil evaluasi model KNN dengan K=10 pada data uji:
# * Akurasi: 95.15% → Model cukup baik dalam memprediksi kelas dengan benar.
# * Precision: 76.92% → Dari semua yang diprediksi sebagai "Miskin", 76% benar.
# * Recall: 83.33% → Dari semua yang seharusnya "Miskin", model hanya menangkap 83.33%.
# * F1 Score: 80.00% → menunjukkan bahwa model memiliki keseimbangan yang baik antara precision (ketepatan klasifikasi positif) dan recall (kemampuan menangkap semua positif).
# * ROC AUC: 90.02% → berarti bahwa model dapat membedakan antara kategori kemiskinan dengan akurasi tinggi
# ----------------------------------------------------------------------

# Feature Importance using Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_importance_df = pd.DataFrame({'Feature': [f'PC{i+1}' for i in range(10)], 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)


# Visualizing Top 10 Important Features
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importance_df['Importance'], y=rf_importance_df['Feature'], palette='coolwarm')
plt.title("Top 10 Fitur Paling Berpengaruh (PCA)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()



# ----------------------------------------------------------------------
# Hasil dari Random Forest Feature Importance menunjukkan bahwa fitur yang paling berpengaruh dalam menentukan klasifikasi kemiskinan adalah:
# * Persentase Kemiskinan (25.47%) → Fitur utama yang sangat menentukan.
# * Pengeluaran per Kapita (9.82%) → Semakin tinggi pengeluaran per kapita, semakin kecil kemungkinan miskin.
# * PDRB (8.84%) → Indikator ekonomi daerah sangat berpengaruh.
# * Indeks Pembangunan Manusia (IPM) (7.75%) → Semakin tinggi IPM, semakin rendah tingkat kemiskinan.
# * Umur Harapan Hidup (4.63%) → Faktor kesehatan juga berpengaruh.
# * Provinsi Papua (4.46%) → Daerah tertentu lebih cenderung masuk kategori miskin.
# * Rata-rata Lama Sekolah (4.25%) → Pendidikan memainkan peran penting dalam kemiskinan.
# * Akses Sanitasi (3.89%) → Infrastruktur kesehatan berpengaruh terhadap kemiskinan.
# * Tingkat Pengangguran (3.50%) → Pengangguran yang tinggi meningkatkan risiko kemiskinan.
# * TPAK (Tingkat Partisipasi Angkatan Kerja) (3.22%) → Semakin tinggi partisipasi tenaga kerja, semakin baik kondisi ekonomi daerah.
# ----------------------------------------------------------------------
