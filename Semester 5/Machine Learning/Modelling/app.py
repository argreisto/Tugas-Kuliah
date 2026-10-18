"""
Aplikasi web sederhana untuk memakai model.joblib dan model_regresi.joblib.
Jalankan: python app.py  lalu buka http://127.0.0.1:5000
(model dibuat dulu lewat python modelling.py)
"""
from pathlib import Path

import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
app = Flask(__name__)

AGE = ["5-14 years", "15-24 years", "25-34 years", "35-54 years", "55-74 years", "75+ years"]
RATE = "suicides/100k pop"

# data dibersihkan dengan cara yang sama seperti di modelling.py
df = pd.read_csv(BASE / "master.csv")
df.columns = df.columns.str.strip()
df = df[df["year"].between(2010, 2012)].copy()
df["gdp_for_year ($)"] = df["gdp_for_year ($)"].astype(str).str.replace(",", "").astype(float)

clf = joblib.load(BASE / "model.joblib")
reg = joblib.load(BASE / "model_regresi.joblib")

# nilai ekonomi per negara dipakai sebagai isian otomatis di form prediksi
ekon = df.groupby("country")[["gdp_per_capita ($)", "gdp_for_year ($)", "HDI for year"]].mean()
pop = df.groupby(["country", "sex", "age"])["population"].mean()

jalur = BASE / "hasil_model.json"
hasil = json.loads(jalur.read_text(encoding="utf-8")) if jalur.exists() else {}
tahun = df.groupby("year")[RATE].mean().round(1).to_dict()

ringkas = {
    "baris": len(df),
    "negara": df["country"].nunique(),
    "rata": round(df[RATE].mean(), 1),
}
jk = df.groupby("sex")[RATE].mean().round(1).to_dict()
umur = df.groupby("age")[RATE].mean().reindex(AGE).round(1).tolist()
top = df.groupby("country")[RATE].mean().sort_values(ascending=False).head(10).round(1)


@app.route("/")
def index():
    return render_template(
        "index.html", r=ringkas, jk=jk, umur=list(zip(AGE, umur)),
        top=list(top.items()), negara=sorted(ekon.index), h=hasil, tahun=tahun,
    )


@app.route("/prediksi", methods=["POST"])
def prediksi():
    d = request.get_json(force=True)
    c, s, a = d.get("country"), d.get("sex"), d.get("age")
    if c not in ekon.index or s not in ("male", "female") or a not in AGE:
        return jsonify(error="Isian belum lengkap"), 400
    e = ekon.loc[c]
    baris = pd.DataFrame([{
        "year": int(d.get("year", 2012)),
        "population": pop.get((c, s, a), df["population"].median()),
        "gdp_per_capita ($)": e["gdp_per_capita ($)"],
        "gdp_for_year ($)": e["gdp_for_year ($)"],
        "HDI for year": e["HDI for year"],
        "country": c, "sex": s, "age": a,
    }])
    p = float(clf.predict_proba(baris)[0][1])
    angka = max(float(reg.predict(baris)[0]), 0)
    return jsonify(risiko="Tinggi" if p >= 0.5 else "Rendah",
                   peluang=round(p * 100), angka=round(angka, 1))


if __name__ == "__main__":
    app.run(debug=True)