import pandas as pd
from scipy.io import arff


def baca_file(path, sep=","):
    """Membaca file csv, xlsx, atau arff jadi tabel pandas."""
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    if path.endswith(".arff"):
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        # arff menyimpan teks dalam bentuk bytes, jadi diubah ke teks biasa
        for k in df.columns:
            df[k] = df[k].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        return df
    return pd.read_csv(path, sep=sep)


def cek(nama, path, label, harapan, sep=","):
    """
    label  : nama kolom label (biner / multi-class)
             ATAU angka jumlah label (multi-label, label ada di kolom paling belakang)
    harapan: BINER / MULTI-CLASS / MULTI-LABEL
    """
    print("\n" + "=" * 55)
    print("Dataset :", nama)

    try:
        df = baca_file(path, sep)
    except FileNotFoundError:
        print("File tidak ditemukan:", path)
        return None

    df.columns = df.columns.astype(str).str.strip()
    print("Jumlah baris :", len(df))
    if len(df) < 250:
        print("PERINGATAN: kurang dari 250 baris!")

    if isinstance(label, int):
        # ---- MULTI-LABEL: ada banyak kolom label berisi 0/1 ----
        kolom_label = list(df.columns[-label:])
        df[kolom_label] = df[kolom_label].astype(int)
        hanya_0_1 = df[kolom_label].isin([0, 1]).all().all()
        baris_ganda = (df[kolom_label].sum(axis=1) > 1).sum()

        print("Jumlah kolom label :", len(kolom_label))
        print("Contoh nama label  :", kolom_label[:5])
        print("Semua isi label hanya 0 dan 1 :", hanya_0_1)
        print("Baris dengan lebih dari 1 label aktif :", baris_ganda)

        if hanya_0_1 and baris_ganda > 0:
            hasil = "MULTI-LABEL"
        else:
            hasil = "BUKAN MULTI-LABEL"
        jumlah = len(kolom_label)
        nama_label = "%d kolom" % len(kolom_label)
    else:
        # ---- BINER / MULTI-CLASS: hanya 1 kolom label ----
        if label not in df.columns:
            print("Kolom '%s' tidak ada. Daftar kolom:" % label)
            print(list(df.columns))
            return None

        print("Kolom label :", label)
        print(df[label].value_counts())
        jumlah = df[label].nunique()
        print("Jumlah kategori label :", jumlah)

        if jumlah == 2:
            hasil = "BINER"
        elif jumlah > 2:
            hasil = "MULTI-CLASS"
        else:
            hasil = "HANYA 1 KATEGORI"
        nama_label = label

    cocok = hasil == harapan
    print("Hasil  :", hasil)
    print("Cocok dengan harapan?", cocok)

    return {
        "Dataset": nama,
        "Kolom label": nama_label,
        "Jumlah label/kategori": jumlah,
        "Jumlah baris": len(df),
        "Hasil": hasil,
        "Cocok": cocok,
    }


hasil_semua = []

# ---------- BINER ----------
hasil_semua.append(cek("Student Depression", "data/student_depression_dataset.csv", "Depression", "BINER"))
hasil_semua.append(cek("Predict the Happiness", "data/train.csv", "Is_Response", "BINER"))
hasil_semua.append(cek("Raisin", "data/Raisin_Dataset.xlsx", "Class", "BINER"))

# ---------- MULTI-CLASS ----------
hasil_semua.append(cek("Date Fruit", "data/Date_Fruit_Datasets.xlsx", "Class", "MULTI-CLASS"))
hasil_semua.append(cek("Maternal Health Risk", "data/Maternal Health Risk Data Set.csv", "RiskLevel", "MULTI-CLASS"))
hasil_semua.append(cek("Student Dropout", "data/data.csv", "Target", "MULTI-CLASS", sep=";"))

# ---------- MULTI-LABEL ----------
hasil_semua.append(cek("Birds", "data/birds.arff", 19, "MULTI-LABEL"))
hasil_semua.append(cek("Genbase", "data/genbase.arff", 27, "MULTI-LABEL"))
hasil_semua.append(cek("Medical", "data/medical.arff", 45, "MULTI-LABEL"))

# ---------- RINGKASAN ----------
print("\n" + "=" * 55)
print("RINGKASAN")
ringkasan = pd.DataFrame([h for h in hasil_semua if h is not None])
print(ringkasan.to_string(index=False))
