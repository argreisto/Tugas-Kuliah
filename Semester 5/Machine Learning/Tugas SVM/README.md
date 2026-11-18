# Tugas Machine Learning: Support Vector Machine (SVM)

- Nama : Argreisto Daniel Krismiawan
- NIM : 241011400910

Repo ini berisi tugas latihan SVM dengan Scikit-learn yang dikerjakan dalam satu notebook. Tiap latihan mengikuti materi yang diberikan, lalu saya tambahkan eksplorasi sendiri (bagian yang judulnya diawali "Tambahan:") supaya hasilnya bisa dibahas lebih dalam.

## Isi notebook

| Latihan | Dataset | Isi utama |
| --- | --- | --- |
| Latihan SKLearn SVM untuk Klasifikasi | `data/diabetes.csv` (Pima Indian Diabetes) | SVC, standarisasi, evaluasi, perbandingan kernel, cross validation, grid search, class_weight, perbandingan dengan model lain |
| Latihan SKLearn SVR | `data/Salary_Data.csv` (lama kerja dan gaji) | SVR rbf, pengaruh C dan gamma, standarisasi target, grid search, repeated cross validation |
| Latihan SVM (Kernel Linear) | data kecil lama belajar dan lama tidur (ada di notebook) | hitungan manual hyperplane dan margin, perbandingan dengan sklearn, dual_coef_, pengaruh C, data pencilan |


## Cara menjalankan

Di komputer sendiri:

```
pip install -r requirements.txt
jupyter notebook Tugas_SVM.ipynb
```

Lalu jalankan semua sel dari atas ke bawah (Run All). Notebook membaca csv dari folder `data/`.

Di Google Colab: upload notebook-nya, lalu upload `diabetes.csv` dan `Salary_Data.csv` saat diminta (atau taruh keduanya di folder `data/`).

Notebook ini sudah dijalankan penuh dan outputnya tersimpan, jadi hasilnya bisa langsung dilihat di GitHub tanpa dijalankan ulang. Saya juga sudah menguji ulang semua sel dari nol di folder bersih, dan hasilnya sama dengan output yang tersimpan.

## Ringkasan hasil

Klasifikasi diabetes:
- SVC default dengan standarisasi: akurasi data uji 75,6% (sama dengan materi). Baseline selalu menebak "tidak diabetes" adalah 66,1%.
- Dari 30 pembagian data yang berbeda akurasinya 71-78% (rata-rata 75,6%), jadi hasil satu split tidak stabil.
- Recall kelas diabetes hanya sekitar 0,57. Dengan `class_weight='balanced'` naik jadi 0,70, tapi akurasi turun ke 71,7%.
- Kernel linear dan rbf setara, dan model lain (Logistic Regression, KNN, Random Forest) juga di kisaran 75-77%.

Regresi gaji (SVR):
- Model dari materi (C=1000, gamma=0,05) kurang bagus: R² latih sekitar 0,35 dan R² cross validation sekitar -0,03.
- Penyebabnya C terlalu kecil dibanding skala gaji. Dengan parameter yang sama tapi target distandarisasi, R² cross validation naik jadi 0,93.
- Grid search menemukan C=100000 dan gamma=0,5 dengan R² data uji 0,875. SVR linear dan Linear Regression hasilnya setara karena datanya hampir linear.

SVM kernel linear:
- Hitungan manual (w1 = w2 = 2/3, b = -17/3, garis batas x2 = -x1 + 8,5) sama dengan hasil sklearn.
- Garis margin yang benar adalah x2 = -x1 + 10 dan x2 = -x1 + 7. Di catatan tertulis +9 dan +8, dan itu tidak cocok dengan support vector-nya, jadi di notebook sudah saya koreksi beserta buktinya.
