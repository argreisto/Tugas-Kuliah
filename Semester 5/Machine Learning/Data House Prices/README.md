# House Prices - Data Cleaning dan Prediksi Harga Rumah

Repository ini berisi proyek machine learning untuk memprediksi harga jual rumah memakai dataset [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dari Kaggle.

Tugas ini dibuat untuk belajar alur pengolahan data sampai model bisa dipakai di aplikasi web sederhana.

Teknologi yang dipakai: Python, Pandas, Scikit-learn, dan Flask.

---

## Dataset

Dataset ada di file `house-prices-advanced-regression-techniques.zip`.

Isinya:

`train.csv` : data latih beserta harga jual (`SalePrice`)

`test.csv` : data uji tanpa harga

`data_description.txt` : penjelasan tiap kolom

`sample_submission.csv` : contoh format hasil prediksi

Ekstrak zip-nya dulu sebelum menjalankan notebook.

---

## Data Cleaning dan EDA

Notebook `Data Cleaning House Prices.ipynb` berisi pengecekan tipe data dan missing value, eksplorasi data, dan persiapan fitur sebelum masuk ke tahap modeling.

Notebook ini bisa dibuka lewat Jupyter Notebook, VS Code, atau Google Colab.

Pastikan path ke `train.csv` sesuai dengan lokasi file di komputermu.

---

## Modelling dan Prediksi

Semua yang berkaitan dengan model ada di folder `modelling/`.

Isinya kode training, model yang sudah disimpan, dan aplikasi web untuk prediksi harga rumah.

Cara instalasi dan menjalankannya ada di [modelling/README.md](modelling/README.md).
