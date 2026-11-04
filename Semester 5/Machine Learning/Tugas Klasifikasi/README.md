# Identifikasi Jenis Klasifikasi pada 9 Dataset (Tugas Machine Learning)

Repository ini berisi kode Python untuk mengecek jenis klasifikasi dari 9 dataset
memakai library **pandas**. Ada 3 jenis klasifikasi, masing-masing 3 dataset.

**Nama:** (isi nama kamu)
**NIM:** (isi NIM kamu)
**Mata kuliah:** Machine Learning

## Penjelasan 3 jenis klasifikasi

| Jenis | Penjelasan singkat | Contoh |
|---|---|---|
| Biner | Label hanya punya 2 kelas. | Sehat / sakit |
| Multi-class | Label punya lebih dari 2 kelas, tapi tiap data hanya boleh masuk 1 kelas. | Jenis bunga: setosa, versicolor, virginica |
| Multi-label | Satu data boleh punya beberapa label sekaligus. Tiap label bernilai 0 atau 1. | Satu film bergenre aksi dan komedi |

## Daftar dataset

Semua dataset diambil dari Kaggle (perlu login Kaggle untuk download).

### Biner
1. Student Depression: https://www.kaggle.com/datasets/hopesb/student-depression-dataset
2. Hotel Review (Predict the Happiness): https://www.kaggle.com/datasets/anu0012/hotel-review
   (pakai `train.csv`, kolom `Is_Response`: happy / not happy)
3. Raisin: https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset

### Multi-class
4. Date Fruit: https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets
5. Maternal Health Risk: https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data
6. Student Dropout & Academic Success: https://www.kaggle.com/datasets/hamnawaseem112222222/predict-students-dropout-and-academic-success
   (3 kelas: dropout, enrolled, graduate)

### Multi-label
7. GoEmotions: https://www.kaggle.com/datasets/debarshichanda/goemotions
8. Toxic Comment (Jigsaw): https://www.kaggle.com/datasets/duaachaseb/multi-label-toxic-comment-classification-dataset
9. Movie Posters (genre film): https://www.kaggle.com/datasets/raman77768/movie-classifier

> Dataset **tidak di-upload** ke GitHub karena ukurannya besar. Download sendiri dari link di atas.

## Cara menjalankan

1. Install library:
   ```
   pip install -r requirements.txt
   ```
2. Buat folder `data/`, lalu taruh file hasil download di dalamnya.
   Kalau ada dua file bernama `train.csv`, ganti namanya jadi `hotel_train.csv` dan `jigsaw_train.csv`.
3. Sesuaikan nama file dan kolom label di bagian bawah `cek_dataset.py` kalau berbeda.
4. Jalankan:
   ```
   python cek_dataset.py
   ```

## Cara kerja pengecekan

- **Biner dan multi-class:** kode menghitung jumlah kelas di kolom label dengan `nunique()`.
  Hasil 2 berarti biner, lebih dari 2 berarti multi-class.
- **Multi-label:** kode mengambil beberapa kolom label paling kanan, lalu mengecek
  (a) semua nilainya hanya 0 atau 1, dan
  (b) ada baris yang punya lebih dari satu label bernilai 1.
  Kalau keduanya benar, berarti multi-label.

## Hasil

(Tempel tabel ringkasan hasil `python cek_dataset.py` di sini setelah kamu jalankan.)

## Referensi

(Tulis jurnal SINTA tahun 2022 sampai sekarang yang sudah kamu baca, format daftar pustaka.)
