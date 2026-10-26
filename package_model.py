"""Part D (2/3): rebuild Part C's split, fit the final pipeline and save it with its helper files."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
threshold = float(config["threshold"])

# the same stratified split as Part C
df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=max_depth,
        min_samples_leaf=int(config["min_samples_leaf"]),
        random_state=42)),
]).fit(X_train, y_train)

# ----------------------------------------------------------- save the artefacts
joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)                  # the three training medians
pd.DataFrame({"column": list(medians), "median": list(medians.values())}).to_csv(
    "models/impute_values.csv", index=False)

# ------------------------------------------------------------- reload check
reloaded = joblib.load("models/delay_model.pkl")
in_memory_prob = model.predict_proba(X_test)[:, 1]
reloaded_prob = reloaded.predict_proba(X_test)[:, 1]
pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
    "value": pd.Series([len(X_test), float(np.abs(in_memory_prob - reloaded_prob).max()), threshold],
                       dtype=object),
}).to_csv("outputs/part_d_reload_check.csv", index=False)

print("saved model; test rows scored:", len(X_test),
      "| max diff:", float(np.abs(in_memory_prob - reloaded_prob).max()))
