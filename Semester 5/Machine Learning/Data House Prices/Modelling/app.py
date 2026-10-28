import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
model = joblib.load(os.path.join(BASE, "model.joblib"))
info = json.load(open(os.path.join(BASE, "hasil_model.json")))


@app.route("/")
def index():
    return render_template("index.html", r=info)


@app.route("/prediksi", methods=["POST"])
def prediksi():
    d = request.get_json(silent=True) or {}
    try:
        row = {
            "OverallQual": int(d["kualitas"]),
            "GrLivArea": float(d["luas"]),
            "GarageCars": int(d["garasi"]),
            "TotalBsmtSF": float(d["basement"]),
            "YearBuilt": int(d["tahun"]),
            "FullBath": int(d["kamar_mandi"]),
            "LotArea": float(d["tanah"]),
            "Neighborhood": str(d["lokasi"]),
        }
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Input tidak lengkap atau tidak valid"), 400
    harga = float(np.expm1(model.predict(pd.DataFrame([row]))[0]))
    return jsonify(harga=round(harga))


if __name__ == "__main__":
    app.run(debug=True)