"""Part D (1/3): hub profile, K-Means segments, separation evidence and the tier check."""

import itertools
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# ------------------------------------------------------------ hub profile
# groupby returns the hubs in ascending hub_id order (HUB01 first, HUB24 last)
profile = df.groupby("hub_id").agg(
    shipments=("hub_id", "size"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index()
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# ------------------------------------------------------ scaling and K-Means
CLUSTER_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[CLUSTER_COLUMNS])
distances = squareform(pdist(scaled))             # Euclidean distance between every pair of hubs

# ------------------------------------------------------ separation evidence
separation_rows = []
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    same, across = [], []
    for i, j in itertools.combinations(range(len(profile)), 2):   # 276 unordered pairs
        (same if labels[i] == labels[j] else across).append(distances[i, j])
    sizes = sorted(np.bincount(labels), reverse=True)
    separation_rows.append({
        "K": k,
        "across_over_within": np.mean(across) / np.mean(same),
        "cluster_sizes": "-".join(str(int(s)) for s in sizes),
    })
pd.DataFrame(separation_rows).round(4).to_csv("outputs/part_d_separation.csv", index=False)

# ----------------------------------------------- K = 3 segments and their names
profile["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)

# name the clusters by the mean delay rate of their hubs, not by the arbitrary cluster number
ranked = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(ranked, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

segments = profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments.to_csv("outputs/hub_segments.csv", index=False)

order = ["Standard", "Watch", "Escalate"]
segment_profile = (profile.groupby("segment")
                   .agg(hubs=("hub_id", "size"),
                        mean_delivery_hours=("mean_delivery_hours", "mean"),
                        delay_rate=("delay_rate", "mean"),
                        mean_distance_km=("mean_distance_km", "mean"),
                        mean_n_stops=("mean_n_stops", "mean"))
                   .loc[order].reset_index())
segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

# ----------------------------------------------- tier label versus behaviour
implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments["segment"] != segments["hub_tier"].map(implied)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(separation_rows)
print(segment_profile.round(4).to_string(index=False))
print(mismatch.to_string(index=False))
