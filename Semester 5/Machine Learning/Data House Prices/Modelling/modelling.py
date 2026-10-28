import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "FullBath", "LotArea"]
CAT = ["Neighborhood"]

df = pd.read_csv("train.csv")
X, y = df[NUM + CAT], np.log1p(df["SalePrice"])
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)


def buat(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("prep", prep), ("model", model)])


kandidat = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=1),
}

hasil = []
for nama, m in kandidat.items():
    p = buat(m).fit(x_tr, y_tr)
    pred = np.expm1(p.predict(x_te))
    aktual = np.expm1(y_te)
    cv = cross_val_score(buat(m), X, y, cv=5, scoring="r2").mean()
    hasil.append({
        "Model": nama,
        "MAE": round(mean_absolute_error(aktual, pred)),
        "RMSE": round(float(np.sqrt(mean_squared_error(aktual, pred)))),
        "R2": round(r2_score(aktual, pred), 3),
        "CV": round(cv, 3),
    })
hasil.sort(key=lambda r: -r["CV"])
terbaik = hasil[0]["Model"]

final = buat(kandidat[terbaik]).fit(X, y)
joblib.dump(final, "model.joblib", compress=3)

korelasi = df.select_dtypes(include="number").corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(8)
info = {
    "baris": len(df),
    "kolom": df.shape[1],
    "rata": round(df["SalePrice"].mean()),
    "median": round(df["SalePrice"].median()),
    "model": hasil,
    "terbaik": terbaik,
    "kualitas": {int(k): round(v) for k, v in df.groupby("OverallQual")["SalePrice"].median().items()},
    "lokasi": [[k, round(v)] for k, v in df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).items()],
    "korelasi": [[k, round(v, 2)] for k, v in korelasi.items()],
    "median_fitur": {c: float(df[c].median()) for c in NUM},
}
json.dump(info, open("hasil_model.json", "w"), indent=1)
print(pd.DataFrame(hasil), "\nModel terbaik:", terbaik)