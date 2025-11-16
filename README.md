# Analisis Sentimen Ulasan Aplikasi Gojek

Proyek ini melakukan **analisis sentimen** pada ulasan pengguna aplikasi Gojek menggunakan **Machine Learning berbasis Python**.  
Model yang digunakan meliputi **Sentence Transformer** untuk embedding teks dan **Logistic Regression** untuk klasifikasi.  

Dataset diambil dari Kaggle: [Gojek Playstore Reviews](https://www.kaggle.com/datasets/dewanakretarta/gojek-playstore-reviews)

## Lisensi Dataset

Dataset ini dirilis di bawah **MIT License**:
# Released under MIT License

Copyright (c) 2013 Mark Otto.

Copyright (c) 2017 Andrew Fong.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Dataset dan Preprocessing
Dataset berisi ulasan aplikasi Gojek dari tahun 2020-2025. Fitur yang digunakan untuk analisis sentimen meliputi:
- `content` : teks ulasan
- `score` : rating bintang (1-5)  

Label sentimen dibagi menjadi:
- **Negatif**: bintang 1-2  
- **Netral**: bintang 3  
- **Positif**: bintang 4-5
  
Dataset asli berukuran besar: **1.093.500 data**. Karena keterbatasan resource, dilakukan **undersampling** menjadi **30.000 data** (10.000 per kelas). 

## Visualisasi distribusi data sebelum undersampling:
<img width="566" height="393" alt="image" src="https://github.com/user-attachments/assets/1f2c416a-e844-478d-9400-2ed53e438617" />

## Visualisasi distribusi data setelah undersampling:
<img width="558" height="393" alt="image" src="https://github.com/user-attachments/assets/e519014c-bf4f-4d7a-a8fe-d54e487343f5" />

Proses yang dilakukan:
1. **Cleaning teks** (menghapus simbol, stopwords, dsb.)  
2. **Embedding teks** menggunakan **Sentence Transformer**  
3. **Klasifikasi** menggunakan **Logistic Regression**

## Pemilihan Metode
- **Sentence Transformer** dipilih karena mampu menghasilkan **representasi vektor semantik** dari kalimat secara efisien. Model ini dapat menangkap konteks dan makna kata secara lebih baik dibanding teknik embedding tradisional seperti TF-IDF, sehingga cocok untuk analisis sentimen berbasis teks yang panjang dan bervariasi.  
- **Logistic Regression** dipilih sebagai classifier karena **sederhana, cepat, dan efektif** untuk masalah klasifikasi multi-class. Logistic Regression juga mudah diinterpretasikan dan cocok untuk dataset berukuran sedang hingga besar setelah embedding.

## Hasil prediksi dan evaluasi model:
<img width="390" height="160" alt="image" src="https://github.com/user-attachments/assets/2ac3d1d6-a2b5-448f-9496-ef67c345ed24" />


# Integrasi dengan FastAPI
Model yang sudah dilatih diintegrasikan dengan aplikasi sederhana berbasis **FastAPI** untuk prediksi real-time.

Karena keterbatasan GitHub untuk file besar, **artifak model** disimpan di Google Drive:  
[Download Artifacts](https://drive.google.com/file/d/1V1djQVSsUhdZ7GJ0XVc2XCyRoay4RBhS/view?usp=sharing)

Isi file zip:
- Folder `artifacts`:
  - `stopword.pkl`
  - `sentiment_classifier.pkl`
- Folder `embedder_model`: model hasil fine-tuning dan token khusus  


**Struktur direktori program:**

![WhatsApp Image 2025-11-16 at 15 34 01_7620dff4](https://github.com/user-attachments/assets/5a7910db-a804-4332-914e-494b0b4e341f)


## Menjalankan API

1. Pastikan artifak model sudah diekstrak sesuai struktur folder.  
2. Jalankan FastAPI:
```bash
uvicorn main:app --reload
```
3. Buka alamat yang tertera (contoh: http://127.0.0.1:8000)
   
<img width="1218" height="183" alt="running fastapi" src="https://github.com/user-attachments/assets/9e146c5f-ff58-4b8e-8b6a-7cb86634d291" />

4. Tambahkan "/docs" untuk membuka Swagger UI: "http://127.0.0.1:8000/docs"

<img width="1351" height="676" alt="image" src="https://github.com/user-attachments/assets/d9a06074-4204-4eea-abfc-06b29331c52c" />

## Prediksi
1. Pilih opsi POST /predict:

<img width="1331" height="605" alt="klik post predict" src="https://github.com/user-attachments/assets/c2979e72-f3cd-4e6b-8bbf-6157b525c933" />

2. Klik Try it Out:

<img width="1331" height="605" alt="try it out" src="https://github.com/user-attachments/assets/264dac74-2019-4b70-b649-719c24885002" />

3. Masukkan komentar pada bagian Body:

<img width="1286" height="532" alt="image" src="https://github.com/user-attachments/assets/486351f7-de70-4072-b41c-c63689f4c12e" />

<img width="1284" height="528" alt="fill text" src="https://github.com/user-attachments/assets/1f05db11-1671-4d2f-afec-fb05e2de8ac5" />

4. Klik Execute:

<img width="1292" height="532" alt="execute" src="https://github.com/user-attachments/assets/30d71836-27e2-49d4-beb0-8c85d6e91d33" />

5. Hasil prediksi akan ditampilkan dalam format JSON:

<img width="1289" height="592" alt="results execute" src="https://github.com/user-attachments/assets/d2088a84-83c3-459b-be55-42644036c3ad" />

Format output menampilkan:
- Teks input
- Hasil cleaning
- Kelas prediksi
- Confidence score per kelas
