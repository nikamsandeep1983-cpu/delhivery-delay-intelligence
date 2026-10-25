"""Part C: delay classifier, feature screening, tuning and the alert threshold."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]


def scores(y_true, y_pred):
    """Confusion-matrix cells plus the four metrics recomputed from those cells."""
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ----------------------------------------------------- 1. stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print("sizes", len(X_train), len(X_val), len(X_test),
      "| delayed", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

# ------------------------------------------ 2. blind baseline + three classifiers
rows = [{"model": "blind_all_on_time", **scores(y_val, np.zeros(len(y_val), dtype=int))}]

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
rows.append({"model": "logistic_regression", **scores(y_val, logreg.predict(X_val))})

best_depth, best_f1 = None, -1.0
for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42).fit(X_train, y_train)
    f1 = scores(y_val, tree.predict(X_val))["f1"]
    if f1 > best_f1:                              # strict ">": ties keep the smaller depth
        best_depth, best_f1 = depth, f1
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42).fit(X_train, y_train)
rows.append({"model": f"decision_tree_depth_{best_depth}", **scores(y_val, tree.predict(X_val))})

forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append({"model": "random_forest_default", **scores(y_val, forest.predict(X_val))})

pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# ------------------------------------------ 3. which columns carry signal?
anova = SelectKBest(f_classif, k="all").fit(X_train, y_train)
feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": anova.scores_,
    "rf_importance": forest.feature_importances_,
})
feature_scores["f_rank"] = feature_scores["f_score"].rank(ascending=False, method="min").astype(int)
feature_scores["is_noise"] = feature_scores["feature"].isin(NOISE_COLUMNS)
feature_scores = feature_scores.sort_values("f_rank")[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
feature_scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": scores(y_val, pipe.predict(X_val))["f1"]})
pd.DataFrame(selectk_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# --------------------------------------------------------------- 4. tuning, k = 7
selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)      # training rows only
X_train_7 = selector.transform(X_train)
X_val_7 = selector.transform(X_val)

param_grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5).fit(X_train_7, y_train)
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,
                          n_iter=9, cv=5, random_state=42).fit(X_train_7, y_train)
default7 = RandomForestClassifier(random_state=42).fit(X_train_7, y_train)


def params_text(params):
    return "; ".join(f"{k}={params[k]}" for k in sorted(params))


tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": params_text(grid.best_params_), "best_cv_accuracy": grid.best_score_,
     "val_f1": scores(y_val, grid.best_estimator_.predict(X_val_7))["f1"]},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": params_text(rand.best_params_), "best_cv_accuracy": rand.best_score_,
     "val_f1": scores(y_val, rand.best_estimator_.predict(X_val_7))["f1"]},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "defaults", "best_cv_accuracy": np.nan,
     "val_f1": scores(y_val, default7.predict(X_val_7))["f1"]},
])
tuning.round(4).to_csv("outputs/part_c_tuning.csv", index=False)

# -------------------------------------------- 5. choose the final configuration
tie_order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
chosen_from = max(tie_order, key=lambda name: (tuning.set_index("search").loc[name, "val_f1"],
                                               -tie_order.index(name)))
chosen_params = {"default_forest_same_7_features": {},
                 "grid_search": grid.best_params_,
                 "randomized_search": rand.best_params_}[chosen_from]

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
]).fit(X_train, y_train)

# ----------------------------------------------------------- 6. alert threshold
val_prob = final_model.predict_proba(X_val)[:, 1]
threshold_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:        # literal list, not np.arange
    threshold_rows.append({"threshold": t, **scores(y_val, (val_prob >= t).astype(int))})
thresholds = pd.DataFrame(threshold_rows).round(4)
thresholds.to_csv("outputs/part_c_thresholds.csv", index=False)

# operations rule: highest threshold whose validation recall is at least 0.80 (else 0.1)
eligible = thresholds[thresholds["recall"] >= 0.80]
chosen_threshold = float(eligible["threshold"].max()) if len(eligible) else 0.1

# ------------------------------------------------- 7. one test evaluation
test_prob = final_model.predict_proba(X_test)[:, 1]
final_test = {"threshold": chosen_threshold, **scores(y_test, (test_prob >= chosen_threshold).astype(int))}
pd.DataFrame([final_test]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)

model_params = final_model.named_steps["model"].get_params()
pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": model_params["n_estimators"],
    "max_depth": "unlimited" if model_params["max_depth"] is None else model_params["max_depth"],
    "min_samples_leaf": model_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}]).to_csv("outputs/part_c_final_config.csv", index=False)

print("chosen:", chosen_from, "| threshold:", chosen_threshold)
