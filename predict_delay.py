"""Part D (3/3): score the 48 new parcels and turn the probabilities into an action list."""

import os

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segments = pd.read_csv("outputs/hub_segments.csv")

model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute["column"], impute["median"].astype(float)))   # training medians, not new ones

# the SAME cleaning and feature functions that built the training table
clean, _ = clean_shipments(new, medians=medians)
features = build_features(clean, hubs)

probability = model.predict_proba(features[feature_order])[:, 1]

result = features[["shipment_id", "hub_id"]].copy()
result["delay_probability"] = probability.round(4)
result["flagged"] = (result["delay_probability"] >= threshold).astype(int)
result = result.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")   # keeps file order


def choose_action(row):
    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"
    if row["flagged"] == 1:
        return "Proactive customer call"
    return "No action"


result["action"] = result.apply(choose_action, axis=1)
result[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]].to_csv(
    "outputs/new_shipment_actions.csv", index=False)

print("parcels:", len(result), "| flagged:", int(result["flagged"].sum()),
      "| escalations:", int((result["action"] == "Escalate to hub manager").sum()))
