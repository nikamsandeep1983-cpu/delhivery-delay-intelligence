"""Part B: delivery-hours regression. Split first, scale without leakage, compare four models."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]


def metrics(y_true, y_pred):
    """MAE, MSE, RMSE and R2 for one set of predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


# ----------------------------------------------------- 1. split 840 / 180 / 180
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ------------------------------------------------- 2. scaling without leakage
scaler = StandardScaler().fit(X_train)            # fitted on the training rows ONLY
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)

pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows",
               "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": pd.Series([len(X_train), len(X_val), len(X_test),
                        round(float(np.abs(train_scaled.mean(axis=0)).max()), 4),
                        round(float(np.abs(test_scaled.mean(axis=0)).max()), 4)], dtype=object),
}).to_csv("outputs/part_b_checks.csv", index=False)

# ------------------------------------------------------------------ 3. baseline
baseline_pred = np.full(len(y_val), y_train.mean())
rows = [{"model": "mean_baseline", "split": "val", **metrics(y_val, baseline_pred)}]

# ------------------------------------------------ 4. tree-depth diagnostic
depth_rows = []
best_depth, best_val_rmse = None, np.inf
for depth in [1, 2, 3, 4, 6, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))
    val_rmse = np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))
    depth_rows.append({"max_depth": "unlimited" if depth is None else depth,
                       "train_rmse": train_rmse, "val_rmse": val_rmse})
    if val_rmse < best_val_rmse:                  # strict "<": ties keep the smaller depth
        best_depth, best_val_rmse = depth, val_rmse

pd.DataFrame(depth_rows).round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# ------------------------------------------------------- 5. four candidate models
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    val_scores[name] = metrics(y_val, model.predict(X_val))
    rows.append({"model": name, "split": "val", **val_scores[name]})

# ------------------------------------------------- 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    m = metrics(y_val, gb.predict(X_val))
    lr_rows.append({"learning_rate": lr, "val_rmse": m["RMSE"], "val_r2": m["R2"]})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# ------------------------------------- 7. cross-validation of the two best models
ranked = sorted(val_scores, key=lambda name: val_scores[name]["RMSE"])
top_two = ranked[:2]

X_all = pd.concat([X_train, X_val])               # 1,020 training-plus-validation rows
y_all = pd.concat([y_train, y_val])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)   # ONE object shared by both models

cv_rows = []
for name in top_two:
    scores = cross_val_score(models[name], X_all, y_all, cv=kfold)   # default scoring = R2
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

# ---------------------------------------- 8. final model, scored on test ONCE
final_name = ranked[0]                            # lowest validation RMSE
rows.append({"model": final_name, "split": "test",
             **metrics(y_test, models[final_name].predict(X_test))})

pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)

print("best depth:", best_depth, "| ranking by val RMSE:", ranked, "| final model:", final_name)
