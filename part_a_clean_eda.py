"""Part A: clean the raw export, build the modelling table and the first delay evidence."""

import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# "Before" counts are measured after removing duplicates and before any filling
dedup = raw.drop_duplicates()

clean, medians = clean_shipments(raw)
feat = build_features(clean, hubs)
feat["is_delayed"] = (feat["delivery_hours"] > feat["promised_hours"]).astype(int)

feat.to_csv("data/modelling_table.csv", index=False)
print("modelling table:", feat.shape)

# ---------------------------------------------------------------- checks file
hours = feat["delivery_hours"]
q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr   # bounds from the UNROUNDED quartiles

checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(dedup["parcel_weight_kg"].isna().sum()),
    "sentinel_weight_rows": int((dedup["parcel_weight_kg"] > 100).sum()),
    "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": int(dedup["service_type"].nunique()),
    "service_type_spellings_after": int(feat["service_type"].nunique()),
    "declared_value_mean_inr": feat["declared_value_inr"].mean(),
    "declared_value_median_inr": feat["declared_value_inr"].median(),
    "log_declared_value_mean": feat["log_declared_value"].mean(),
    "log_declared_value_median": feat["log_declared_value"].median(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(feat),
    "nulls_in_features_and_target": int(feat[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(feat["is_delayed"].sum()),
    "delay_rate": feat["is_delayed"].mean(),
    "hours_q1": q1,
    "hours_q3": q3,
    "hours_lower_bound": lower,
    "hours_upper_bound": upper,
    "hours_outlier_count": int(((hours < lower) | (hours > upper)).sum()),
}
values = [round(v, 4) if isinstance(v, (float, np.floating)) else int(v) for v in checks.values()]
pd.DataFrame({"metric": list(checks), "value": pd.Series(values, dtype=object)}).to_csv(
    "outputs/part_a_checks.csv", index=False)

# ------------------------------------------------------------ delay by group
feat["distance_band"] = pd.cut(feat["distance_km"], bins=[0, 200, 400, 600, 2000],
                               labels=["0-200", "201-400", "401-600", "601+"],
                               right=True, include_lowest=True)

parts = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = feat.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index().rename(columns={dim: "group"})
    g["delay_rate"] = g["delayed"] / g["shipments"]
    g["group"] = g["group"].astype(str)
    g.insert(0, "dimension", dim)
    parts.append(g[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]])
by_group = pd.concat(parts, ignore_index=True)
by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# ------------------------------------------------------------- correlations
corr = feat.select_dtypes("number").corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv", index=True)   # row labels kept as first column

# helper for the README: three columns most correlated with delivery_hours
top3 = corr["delivery_hours"].drop("delivery_hours").abs().sort_values(ascending=False).head(3).index
print(corr.loc[top3, "delivery_hours"])
