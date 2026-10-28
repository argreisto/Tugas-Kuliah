# Analisis dan Prediksi Harga Rumah

Tugas analisis data: dataset [House Prices (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

- `data_cleaning_house_prices.ipynb`: cleaning, EDA, splitting, modelling
- `modelling.py`: latih 3 model, simpan `model.joblib` dan `hasil_model.json`
- `app.py` + `templates/index.html`: web Flask

## Menjalankan
```
pip install -r requirements.txt
python modelling.py
python app.py
```
Deploy (Render): Start Command `gunicorn app:app`.
