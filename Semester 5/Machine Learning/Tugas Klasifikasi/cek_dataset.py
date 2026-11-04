import os
import pandas as pd


def baca(path, sep=","):
    """Baca file sesuai ekstensinya (.xlsx, .tsv, atau .csv)."""
    if path.endswith(".xlsx"):
        return pd.read_excel(path)  # butuh openpyxl
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=sep)


def cek(nama, path, label, jenis, sep=","):
    """
    nama  : nama dataset (untuk ditampilkan)
    path  : lokasi file
    label : untuk BINER / MULTI-CLASS -> nama kolom label
            untuk MULTI-LABEL         -> jumlah kolom label (dihitung dari kolom paling kanan)
    jenis : jenis yang diharapkan ("BINER", "MULTI-CLASS", "MULTI-LABEL")
    """
    print("=" * 55)
    print(nama, "| harapan:", jenis)

    if not os.path.exists(path):
        print("File tidak ditemukan:", path)
        return {"dataset": nama, "harapan": jenis, "hasil": "FILE HILANG", "cocok": False}

    df = baca(path, sep)
    print("Jumlah baris:", len(df))

    if jenis in ("BINER", "MULTI-CLASS"):
        # Satu kolom label, hitung ada berapa kelas
        jumlah = df[label].nunique()
        print(df[label].value_counts())
        if jumlah == 2:
            hasil = "BINER"
        elif jumlah > 2:
            hasil = "MULTI-CLASS"
        else:
            hasil = "HANYA 1 KELAS"
        print("Jumlah kelas:", jumlah, "->", hasil)
    else:
        # Beberapa kolom label (0/1). Ambil kolom paling kanan.
        kolom = df.columns[-label:]
        print("Jumlah kolom label:", len(kolom))
        cuma_01 = all(set(df[k].dropna().unique()) <= {0, 1} for k in kolom)
        print("Semua label berisi 0/1 saja?", cuma_01)
        banyak = int((df[kolom].sum(axis=1) > 1).sum())
        print("Baris dengan lebih dari 1 label aktif:", banyak)
        if len(kolom) > 2 and cuma_01 and banyak > 0:
            hasil = "MULTI-LABEL"
        else:
            hasil = "BUKAN MULTI-LABEL"
        print("->", hasil)

    cocok = hasil == jenis
    print("COCOK!" if cocok else "TIDAK COCOK, cek lagi")
    return {"dataset": nama, "harapan": jenis, "hasil": hasil, "cocok": cocok}


def cek_tsv_indeks(nama, path, jenis="MULTI-LABEL"):
    """
    Khusus GoEmotions versi train.tsv.
    Isi file: tiga kolom tanpa judul -> teks, daftar nomor emosi (misal "3,27"), id komentar.
    Jadi labelnya bukan kolom 0/1, tapi nomor emosi yang dipisah koma.
    """
    print("=" * 55)
    print(nama, "| harapan:", jenis)

    if not os.path.exists(path):
        print("File tidak ditemukan:", path)
        return {"dataset": nama, "harapan": jenis, "hasil": "FILE HILANG", "cocok": False}

    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label", "id"])
    print("Jumlah baris:", len(df))

    # pecah "3,27" jadi daftar nomor
    daftar = df["label"].astype(str).str.split(",")
    jumlah_per_baris = daftar.apply(len)
    nomor_unik = set(n for baris in daftar for n in baris)

    print("Jumlah nomor emosi yang berbeda:", len(nomor_unik))
    print("Baris dengan lebih dari 1 label:", int((jumlah_per_baris > 1).sum()))
    print("Paling banyak label dalam 1 baris:", int(jumlah_per_baris.max()))

    if len(nomor_unik) > 2 and (jumlah_per_baris > 1).sum() > 0:
        hasil = "MULTI-LABEL"
    else:
        hasil = "BUKAN MULTI-LABEL"
    print("->", hasil)

    cocok = hasil == jenis
    print("COCOK!" if cocok else "TIDAK COCOK, cek lagi")
    return {"dataset": nama, "harapan": jenis, "hasil": hasil, "cocok": cocok}


hasil_semua = []

# ---------- BINER ----------
hasil_semua.append(cek("Student Depression", "data/Student Depression Dataset.csv", "Depression", "BINER"))
hasil_semua.append(cek("Hotel Review (Happiness)", "data/hotel_train.csv", "Is_Response", "BINER"))
hasil_semua.append(cek("Raisin", "data/Raisin_Dataset.xlsx", "Class", "BINER"))

# ---------- MULTI-CLASS ----------
hasil_semua.append(cek("Date Fruit", "data/Date_Fruit_Datasets.xlsx", "Class", "MULTI-CLASS"))
hasil_semua.append(cek("Maternal Health Risk", "data/Maternal Health Risk Data Set.csv", "RiskLevel", "MULTI-CLASS"))
# Kalau kolom menyatu jadi satu, pakai pemisah titik koma (sep=";")
hasil_semua.append(cek("Student Dropout", "data/data.csv", "Target", "MULTI-CLASS", sep=";"))

# ---------- MULTI-LABEL ----------
hasil_semua.append(cek_tsv_indeks("GoEmotions", "data/train.tsv"))
hasil_semua.append(cek("Jigsaw Toxic", "data/jigsaw_train.csv", 6, "MULTI-LABEL"))
hasil_semua.append(cek("Movie Posters", "data/movie_train.csv", 25, "MULTI-LABEL"))

# ---------- RINGKASAN ----------
print("\n" + "=" * 55)
print("RINGKASAN")
print(pd.DataFrame(hasil_semua).to_string(index=False))
