# Analisis Tingkat Suicide Tahun 2010-2012

Tugas analisis dan prediksi tingkat suicide (per 100 ribu penduduk) memakai klasifikasi, regresi, dan clustering.

- Nama : Argreisto Daniel Krismiawan
- NIM : 241011400910
- Kelas : 05TPLP023

## Dataset
Suicide Rates Overview 1985 to 2016 (`master.csv`) dari Kaggle:
https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016

Data difilter ke tahun 2010-2012, hasilnya 3.060 baris dari 92 negara.

## Isi repo
| File | Keterangan |
| --- | --- |
| `Analisis_Suicide_Rates.ipynb` | Notebook lengkap: EDA, penjelasan, dan hasil |
| `modelling.py` | Script modelling (cleaning, split, klasifikasi, regresi, clustering, simpan model) |
| `master.csv` | Dataset |
| `model.joblib` | Pipeline klasifikasi terbaik (preprocessing + model) |
| `model_regresi.joblib` | Pipeline regresi terbaik |
| `requirements.txt` | Library yang dibutuhkan |

## Cara menjalankan
```bash
pip install -r requirements.txt
python modelling.py
```
Setelah selesai, file `model.joblib` dan `model_regresi.joblib` akan dibuat ulang di folder yang sama.

## Cara memakai model
```python
import joblib
import pandas as pd

model = joblib.load("model.joblib")

data_baru = pd.DataFrame([{
    "year": 2012, "population": 500000, "gdp_per_capita ($)": 15000,
    "gdp_for_year ($)": 8000000000, "HDI for year": 0.75,
    "country": "Indonesia", "sex": "male", "age": "35-54 years",
}])
print(model.predict(data_baru))  # 0 = Rendah, 1 = Tinggi
```
Catatan: `country` harus salah satu negara yang ada di data latih, dan versi scikit-learn sebaiknya sama dengan saat model disimpan.

## Model
- Klasifikasi (risiko Tinggi/Rendah): Logistic Regression, Decision Tree, KNN, Random Forest
- Regresi: Linear Regression, Decision Tree, KNN, Random Forest
- Clustering: K-Means dan Hierarchical (Ward), K = 4
- Evaluasi tambahan: split per negara (GroupShuffleSplit) supaya model dites ke negara yang benar-benar baru

## Menjalankan tampilan web
Setelah `model.joblib` dan `model_regresi.joblib` ada:
```bash
python app.py
```
Lalu buka http://127.0.0.1:5000 di browser (paling pas dilihat dalam ukuran layar HP).
