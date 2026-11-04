# Tugas Machine Learning: Klasifikasi Biner, Multi-class, dan Multi-label

Nama  : Argreisto Daniel Krismiawan

NIM   : 241011400910

Kelas : 05TPLP023

## Tentang tugas ini

Di tugas ini saya diminta mencari 3 jenis klasifikasi, yaitu biner, multi-class, dan multi-label. Tiap jenis saya cari 3 dataset, jadi totalnya 9 dataset. Setelah itu saya cek pakai library pandas apakah label di tiap dataset memang sesuai jenisnya, misalnya benar atau tidak labelnya cuma 2 atau lebih dari 2.

Semua kodenya ada di file cek_dataset.py.

## Yang saya pelajari

Awalnya saya kira ketiganya sama saja, ternyata beda.

Klasifikasi biner itu labelnya cuma ada 2 pilihan. Contohnya senang atau tidak senang.

Klasifikasi multi-class labelnya lebih dari 2 pilihan, tapi satu data cuma boleh masuk ke satu pilihan saja. Contohnya tingkat risiko: rendah, sedang, tinggi. Seorang ibu hamil hanya masuk salah satunya.

Klasifikasi multi-label itu satu data boleh punya beberapa label sekaligus. Contohnya satu film bisa bergenre aksi dan komedi. Di dataset multi-label, tiap label biasanya jadi satu kolom sendiri yang isinya 0 atau 1. Jadi tiap kolom itu sebenarnya biner, tapi karena kolomnya banyak dan satu baris boleh punya lebih dari satu angka 1, jadinya multi-label.

Jadi gampangnya, biner dan multi-class cuma punya satu kolom label, sedangkan multi-label punya banyak kolom label.

## Dataset yang saya pakai

Semuanya dari Kaggle dan perlu login untuk download. Datasetnya tidak saya upload ke GitHub karena ukurannya besar (terutama Hotel Review dan GoEmotions), jadi silakan download lewat link di bawah.

Biner:
1. Student Depression, kolom label Depression (0 dan 1)
   https://www.kaggle.com/datasets/hopesb/student-depression-dataset
2. Hotel Review (Predict the Happiness), kolom label Is_Response (happy dan not happy), pakai file train.csv
   https://www.kaggle.com/datasets/anu0012/hotel-review
3. Raisin, kolom label Class (2 varietas kismis)
   https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset

Multi-class:
4. Date Fruit, kolom label Class (7 jenis kurma)
   https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets
5. Maternal Health Risk, kolom label RiskLevel (low, mid, high)
   https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data
6. Student Dropout and Academic Success, kolom label Target (dropout, enrolled, graduate)
   https://www.kaggle.com/datasets/hamnawaseem112222222/predict-students-dropout-and-academic-success

Multi-label:
7. GoEmotions, 28 kolom emosi (27 emosi ditambah netral)
   https://www.kaggle.com/datasets/debarshichanda/goemotions
8. Toxic Comment Classification (Jigsaw, versi Emoji-Aware), 6 kolom jenis toxic
   https://www.kaggle.com/datasets/duaachaseb/multi-label-toxic-comment-classification-dataset
9. Movie Posters, 25 kolom genre film (pakai file csv di folder Multi_Label_dataset)
   https://www.kaggle.com/datasets/raman77768/movie-classifier

## Cara ngecek pakai pandas

Untuk biner dan multi-class, saya hitung jumlah kelas di kolom label pakai nunique(). Kalau hasilnya 2 berarti biner, kalau lebih dari 2 berarti multi-class. Saya juga tampilkan value_counts() supaya kelihatan jumlah data tiap kelasnya.

    jumlah = df["Class"].nunique()
    print(df["Class"].value_counts())

Untuk multi-label, saya ambil kolom-kolom label di paling kanan. Terus saya cek dua hal: apakah isinya cuma 0 dan 1, dan apakah ada baris yang punya lebih dari satu label bernilai 1. Kalau dua-duanya iya, berarti benar multi-label.

    kolom = df.columns[-28:]
    banyak = (df[kolom].sum(axis=1) > 1).sum()

## Cara menjalankan

1. Install library dulu: pip install -r requirements.txt
2. Download dataset dari link di atas. Saya taruh semuanya di folder Dataset, satu dataset satu subfolder (contoh: Dataset/Raisin_Dataset, Dataset/GoEmotions, dst). Folder Dataset ini sejajar dengan cek_dataset.py. Kode akan mencari file-nya sendiri sampai ke dalam subfolder.
3. Jalankan dari dalam folder repository: python cek_dataset.py
4. Kalau mau disimpan ke file: python cek_dataset.py > hasil.txt

Folder Dataset tidak saya upload ke GitHub karena ukurannya besar, makanya masuk .gitignore.

## Hasil

Output lengkap ada di file hasil.txt. Ringkasannya begini.

Dataset biner:
- Student Depression: 27.901 baris, kolom Depression punya 2 nilai (1 = 16.336, 0 = 11.565). Jadi biner.
- Hotel Review: 38.932 baris, kolom Is_Response punya 2 kelas (happy 26.521, not happy 12.411). Jadi biner.
- Raisin: 900 baris, kolom Class punya 2 kelas (Kecimen 450, Besni 450). Jadi biner.

Dataset multi-class:
- Date Fruit: 898 baris, kolom Class punya 7 kelas (Dokol 204, Safavi 199, Rotana 166, Deglet 98, Sogay 94, Iraqi 72, Berhi 65). Jadi multi-class.
- Maternal Health Risk: 1.014 baris, kolom RiskLevel punya 3 kelas (low risk 406, mid risk 336, high risk 272). Jadi multi-class.
- Student Dropout: 4.424 baris, kolom Target punya 3 kelas (Graduate 2.209, Dropout 1.421, Enrolled 794). Jadi multi-class.

Dataset multi-label:
- GoEmotions: 70.000 baris (file goemotions_1.csv), 28 kolom emosi (27 emosi dan neutral) yang isinya cuma 0 dan 1. Ada 11.887 baris yang punya lebih dari satu label aktif. Jadi multi-label.
- Jigsaw Toxic (Emoji-Aware Toxic Comment): 22.303 baris, 6 kolom label (toxic, severe_toxic, obscene, threat, insult, identity_hate) yang isinya cuma 0 dan 1. Ada 11.859 baris yang punya lebih dari satu label aktif. Jadi multi-label.
- Movie Posters: 7.254 baris, 25 kolom genre (Action, Adventure, Animation, dan seterusnya) yang isinya cuma 0 dan 1. Ada 5.932 baris yang punya lebih dari satu genre. Jadi multi-label.

Tabel ringkasan dari program:

    dataset                   harapan      hasil        cocok
    Student Depression        BINER        BINER        True
    Hotel Review (Happiness)  BINER        BINER        True
    Raisin                    BINER        BINER        True
    Date Fruit                MULTI-CLASS  MULTI-CLASS  True
    Maternal Health Risk      MULTI-CLASS  MULTI-CLASS  True
    Student Dropout           MULTI-CLASS  MULTI-CLASS  True
    GoEmotions                MULTI-LABEL  MULTI-LABEL  True
    Jigsaw Toxic              MULTI-LABEL  MULTI-LABEL  True
    Movie Posters             MULTI-LABEL  MULTI-LABEL  True

Kesimpulannya, label di ke-9 dataset sudah sesuai dengan jenis klasifikasi yang saya pilih. Tiga dataset pertama biner karena kelasnya ada 2. Tiga berikutnya multi-class karena kelasnya lebih dari 2 dan tiap data hanya punya satu kelas. Tiga terakhir multi-label karena labelnya banyak kolom 0/1 dan ada baris yang punya lebih dari satu label aktif.

Catatan kecil: di dataset Movie Posters ada satu kolom genre bernama "N/A", artinya film yang genrenya tidak diketahui. Kolom itu tetap saya hitung sebagai salah satu dari 25 kolom label.

## Referensi

Siringoringo, R., Jamaluddin, & Perangin-angin, R. (2022). Text mining dan klasifikasi multi label menggunakan XGBoost. METHOMIKA: Jurnal Manajemen Informatika & Komputerisasi Akuntansi, 6(2), 234-238. https://doi.org/10.46880/jmika.Vol6No2.pp234-238

Dari artikel ini saya belajar bahwa klasifikasi biasa cuma mencari satu label, sedangkan multi-label memungkinkan satu teks punya beberapa label. Penulisnya memakai data artikel ilmiah dengan 6 kategori (Computer Science, Physics, Mathematics, Statistics, Quantitative Biology, Quantitative Finance), dan satu artikel bisa masuk ke lebih dari satu kategori. Mereka membandingkan XGBoost dengan Logistic Regression. Hasilnya XGBoost lebih bagus, rata-rata akurasi 0,88 dan F1-score 0,78, sedangkan Logistic Regression akurasi 0,81 dan F1-score 0,47. Dari sini saya jadi lebih paham bahwa di multi-label tiap kolom label berisi 0 atau 1 dan satu baris boleh punya lebih dari satu angka 1, sama seperti dataset GoEmotions, Jigsaw, dan Movie Posters yang saya pakai.
