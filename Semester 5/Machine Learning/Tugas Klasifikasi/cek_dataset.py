import os
import pandas as pd

# Supaya file dibaca dari folder tempat script ini berada,
# walaupun terminal sedang terbuka di folder lain.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Folder induk dataset: pakai data/ atau Dataset/ kalau ada, kalau tidak pakai folder script ini.
FOLDER = next((f for f in ("data", "Dataset", "dataset") if os.path.isdir(f)), ".")

EKSTENSI = (".csv", ".xlsx", ".xls", ".tsv")


def cari(awalan, sub=""):
    """
    Cari file data yang namanya diawali 'awalan' (boleh beberapa pilihan dalam tuple).
    'sub' = potongan nama subfolder dataset-nya, supaya tidak tertukar
    dengan file lain yang namanya sama (misalnya dua file train.csv).
    """
    if isinstance(awalan, str):
        awalan = (awalan,)
    awalan = tuple(a.lower() for a in awalan)

    def telusuri(pakai_sub):
        for root, _, files in os.walk(FOLDER):
            if pakai_sub and sub.lower() not in root.lower():
                continue
            for nama in sorted(files):
                n = nama.lower()
                if n.startswith(awalan) and n.endswith(EKSTENSI):
                    return os.path.join(root, nama)
        return None

    return telusuri(True) or telusuri(False)


def baca(path):
    """Baca file sesuai ekstensinya. Pemisah csv (koma / titik koma) dicek otomatis."""
    kecil = path.lower()
    if kecil.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)  # butuh openpyxl
    if kecil.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    with open(path, encoding="utf-8", errors="ignore") as f:
        baris_pertama = f.readline()
    sep = ";" if baris_pertama.count(";") > baris_pertama.count(",") else ","
    return pd.read_csv(path, sep=sep)


def cek(nama, awalan, label, jenis, sub=""):
    """
    nama   : nama dataset (untuk ditampilkan)
    awalan : awal nama file di komputer (tanpa perlu menulis ekstensi)
    sub    : potongan nama subfolder dataset (opsional)
    label  : BINER / MULTI-CLASS -> nama kolom label
             MULTI-LABEL         -> jumlah kolom label (dihitung dari kolom paling kanan)
    jenis  : jenis yang diharapkan ("BINER", "MULTI-CLASS", "MULTI-LABEL")
    """
    print("=" * 55)
    print(nama, "| harapan:", jenis)

    path = cari(awalan, sub)
    if path is None:
        print("File tidak ditemukan, nama diawali:", awalan)
        return {"dataset": nama, "harapan": jenis, "hasil": "FILE HILANG", "cocok": False}

    print("File:", path)
    df = baca(path)
    print("Jumlah baris:", len(df))

    if jenis in ("BINER", "MULTI-CLASS"):
        # Satu kolom label, hitung ada berapa kelas
        if label not in df.columns:
            print("Kolom", label, "tidak ada. Kolom yang ada:", list(df.columns))
            return {"dataset": nama, "harapan": jenis, "hasil": "KOLOM SALAH", "cocok": False}
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
        print("Kolom label:", list(kolom))
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


hasil_semua = []

# ---------- BINER ----------
hasil_semua.append(cek("Student Depression", "Student Depression Dataset", "Depression", "BINER", sub="Student_Depression"))
hasil_semua.append(cek("Hotel Review (Happiness)", "hotel_train", "Is_Response", "BINER", sub="Hotel"))
hasil_semua.append(cek("Raisin", "Raisin_Dataset", "Class", "BINER", sub="Raisin"))

# ---------- MULTI-CLASS ----------
hasil_semua.append(cek("Date Fruit", "Date_Fruit_Datasets", "Class", "MULTI-CLASS", sub="Date_Fruit"))
hasil_semua.append(cek("Maternal Health Risk", "Maternal Health Risk", "RiskLevel", "MULTI-CLASS", sub="Maternal"))
hasil_semua.append(cek("Student Dropout", ("student_dropout", "data"), "Target", "MULTI-CLASS", sub="Student_Dropout"))

# ---------- MULTI-LABEL ----------
hasil_semua.append(cek("GoEmotions", "goemotions_1", 28, "MULTI-LABEL", sub="GoEmotions"))
hasil_semua.append(cek("Jigsaw Toxic", ("Emoji", "train", "jigsaw"), 6, "MULTI-LABEL", sub="Toxic"))
hasil_semua.append(cek("Movie Posters", "train.csv", 25, "MULTI-LABEL", sub="Multi_Label"))

# ---------- RINGKASAN ----------
print("\n" + "=" * 55)
print("RINGKASAN")
print(pd.DataFrame(hasil_semua).to_string(index=False))
