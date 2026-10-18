"""
Modelling: Analisis Tingkat Suicide Tahun 2010-2012
Data   : Kaggle - Suicide Rates Overview 1985 to 2016 (master.csv)
Isi    : loading, cleaning, split, preprocessing, klasifikasi, regresi,
         evaluasi split per negara, clustering, lalu simpan model (.joblib)
"""
from pathlib import Path

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import (GroupShuffleSplit, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BASE = Path(__file__).resolve().parent
SEED = 1


# ---------------------------------------------------------------- 1. Data
raw = pd.read_csv(BASE / "master.csv")
raw.columns = raw.columns.str.strip()
df = raw[(raw["year"] >= 2010) & (raw["year"] <= 2012)].copy()
print("Ukuran data asli:", raw.shape, "| setelah filter 2010-2012:", df.shape)


# ------------------------------------------------------------- 2. Cleaning
df["gdp_for_year ($)"] = df["gdp_for_year ($)"].astype(str).str.replace(",", "").astype(float)
df = df.drop(columns=["country-year"]).drop_duplicates()
print("Ukuran setelah cleaning:", df.shape)


# ---------------------------------------------------------------- 3. Split
num_cols = ["year", "population", "gdp_per_capita ($)", "gdp_for_year ($)", "HDI for year"]
nom_cols = ["country", "sex"]
ord_cols = ["age"]
order = ["5-14 years", "15-24 years", "25-34 years", "35-54 years", "55-74 years", "75+ years"]

X = df[num_cols + nom_cols + ord_cols]
y_reg = df["suicides/100k pop"]

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=SEED)

# label klasifikasi: di atas median data latih = Tinggi (1), selain itu Rendah (0)
median_train = y_reg_train.median()
y_train = (y_reg_train > median_train).astype(int)
y_test = (y_reg_test > median_train).astype(int)
print("Median train:", round(median_train, 2))


# --------------------------------------------------------- 4. Preprocessing
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("nom", OneHotEncoder(handle_unknown="ignore"), nom_cols),
    ("ord", OrdinalEncoder(categories=[order]), ord_cols),
])


# ------------------------------------------------------- 5. Klasifikasi
print("\n=== KLASIFIKASI ===")
cv_knn = {}
for k in [3, 5, 7, 9, 11, 15]:
    pipe = Pipeline([("prep", prep), ("m", KNeighborsClassifier(n_neighbors=k))])
    cv_knn[k] = cross_val_score(pipe, X_train, y_train, cv=5).mean()
    print(f"K = {k} | akurasi CV = {cv_knn[k]:.3f}")
best_k = max(cv_knn, key=cv_knn.get)
print("K terbaik:", best_k)

clf_models = {
    "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=best_k),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
}

rows, fitted = [], {}
for name, m in clf_models.items():
    pipe = Pipeline([("prep", prep), ("model", m)]).fit(X_train, y_train)
    p = pipe.predict(X_test)
    fitted[name] = pipe
    rows.append([name, accuracy_score(y_test, p), precision_score(y_test, p),
                 recall_score(y_test, p), f1_score(y_test, p)])

hasil_clf = (pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
             .round(3).sort_values("F1", ascending=False))
print(hasil_clf.to_string(index=False))

best = hasil_clf.iloc[0]["Model"]
print("Model klasifikasi terbaik (F1):", best)


# ----------------------------------------------------------- 6. Regresi
print("\n=== REGRESI ===")
reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=SEED),
    "KNN": KNeighborsRegressor(n_neighbors=best_k),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
}

rows, fitted_reg = [], {}
for name, m in reg_models.items():
    pipe = Pipeline([("prep", prep), ("model", m)]).fit(X_train, y_reg_train)
    p = pipe.predict(X_test)
    fitted_reg[name] = pipe
    rows.append([name, mean_absolute_error(y_reg_test, p),
                 np.sqrt(mean_squared_error(y_reg_test, p)), r2_score(y_reg_test, p)])

hasil_reg = (pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"])
             .round(3).sort_values("R2", ascending=False))
print(hasil_reg.to_string(index=False))

best_r = hasil_reg.iloc[0]["Model"]
print("Model regresi terbaik (R2):", best_r)


# -------------------------------------- 7. Evaluasi lebih ketat: split negara
print("\n=== SPLIT PER NEGARA ===")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(X, y_reg, groups=df["country"]))
Xg_train, Xg_test = X.iloc[tr], X.iloc[te]
yg_reg_train, yg_reg_test = y_reg.iloc[tr], y_reg.iloc[te]
med_g = yg_reg_train.median()
yg_train = (yg_reg_train > med_g).astype(int)
yg_test = (yg_reg_test > med_g).astype(int)
print("Negara train:", df.iloc[tr]["country"].nunique(),
      "| Negara test:", df.iloc[te]["country"].nunique())

rows = []
for name, m in clf_models.items():
    p = Pipeline([("prep", prep), ("model", m)]).fit(Xg_train, yg_train).predict(Xg_test)
    rows.append([name, accuracy_score(yg_test, p), f1_score(yg_test, p)])
print(pd.DataFrame(rows, columns=["Model", "Accuracy", "F1"]).round(3).to_string(index=False))

rows = []
for name, m in reg_models.items():
    p = Pipeline([("prep", prep), ("model", m)]).fit(Xg_train, yg_reg_train).predict(Xg_test)
    rows.append([name, mean_absolute_error(yg_reg_test, p), r2_score(yg_reg_test, p)])
print(pd.DataFrame(rows, columns=["Model", "MAE", "R2"]).round(3).to_string(index=False))


# ----------------------------------------------------------- 8. Clustering
print("\n=== CLUSTERING ===")
neg = (df.groupby("country")
         .agg({"suicides/100k pop": "mean", "gdp_per_capita ($)": "mean", "population": "sum"})
         .reset_index())
neg.columns = ["country", "rate", "gdp_pc", "pop"]
Xc = StandardScaler().fit_transform(neg[["rate", "gdp_pc", "pop"]])

for k in range(2, 9):
    km_k = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xc)
    print(f"K = {k} | silhouette = {silhouette_score(Xc, km_k.labels_):.3f}")

K = 4
km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(Xc)
neg["KMeans"] = km.labels_
print("Silhouette K-Means (K=4):", round(silhouette_score(Xc, km.labels_), 4))

hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Xc)
neg["HC"] = hc.labels_
print("Silhouette Hierarchical (K=4):", round(silhouette_score(Xc, hc.labels_), 4))
print(neg.groupby("HC")[["rate", "gdp_pc", "pop"]].mean().round(1))


# ------------------------------------------------ 9. Simpan hasil dan model
segmen = (neg.groupby("HC").agg(rate=("rate", "mean"), gdp_pc=("gdp_pc", "mean"), n=("country", "count"))
          .round(1).reset_index().to_dict("records"))
hasil = {"klasifikasi": hasil_clf.to_dict("records"),
         "regresi": hasil_reg.to_dict("records"), "segmen": segmen}
(BASE / "hasil_model.json").write_text(json.dumps(hasil, indent=2), encoding="utf-8")

# yang disimpan satu pipeline utuh (preprocessing + model), jadi bisa langsung predict
joblib.dump(fitted[best], BASE / "model.joblib", compress=3)
joblib.dump(fitted_reg[best_r], BASE / "model_regresi.joblib", compress=3)
print("\nModel tersimpan: model.joblib dan model_regresi.joblib")