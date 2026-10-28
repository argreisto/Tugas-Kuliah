# Prediksi Harga Rumah

Tugas ini dipakai untuk memprediksi harga rumah di Ames, Iowa. Datanya ada 1.460 rumah dengan 81 kolom. Hasil modelnya saya tampilkan dalam web sederhana pakai Flask.

## Isi folder Modelling

- `data_cleaning_house_prices.ipynb`: cleaning data, EDA, dan splitting data
- `modelling.py`: melatih model lalu menyimpannya ke `model.joblib` dan `hasil_model.json`
- `app.py`: aplikasi Flask
- `templates/index.html`: tampilan web
- `train.csv`: dataset
- `requirements.txt`: daftar library yang dipakai

## Yang dikerjakan

1. Cleaning data. Nilai kosong diisi median (kolom angka) atau modus (kolom kategori), dan kolom yang datanya terlalu banyak kosong dibuang.
2. EDA untuk melihat sebaran harga dan fitur mana yang paling berhubungan dengan harga.
3. Modelling dengan tiga algoritma: Linear Regression, Random Forest, dan Gradient Boosting.
4. Evaluasi pakai MAE, RMSE, R2, dan cross validation 5 fold.
5. Hasilnya dibuat jadi web dengan halaman Ringkasan, Eksplorasi, Model, dan Prediksi.

## Fitur yang dipakai di web

Supaya form prediksi tidak terlalu panjang, saya hanya pakai 8 fitur: OverallQual, GrLivArea, TotalBsmtSF, LotArea, YearBuilt, GarageCars, FullBath, dan Neighborhood.

Harga dilatih dalam bentuk log (`log1p`) supaya sebarannya lebih rata, lalu dikembalikan ke dolar waktu prediksi.

## Hasil

Model terbaik adalah Gradient Boosting, dipilih dari nilai cross validation (R2 sekitar 0,86). Rata-rata kesalahannya (MAE) sekitar $17.900. Tabel lengkapnya bisa dilihat di halaman Model.

## Cara menjalankan

```
pip install -r requirements.txt
python modelling.py
python app.py
```

Lalu buka http://127.0.0.1:5000 di browser. `modelling.py` harus dijalankan dulu karena file `model.joblib` dibuat dari situ.

