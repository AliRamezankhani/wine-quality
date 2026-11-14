# %%
"""Model selection and hyperparameter tuning (cross-validation on the training part only).

Run:  python model_selection.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_prep import FEATURES, TARGET, load_and_clean, split_data

os.makedirs("results", exist_ok=True)
SEED = 1

# %%
df = load_and_clean()
df_full_train, df_test = split_data(df)
X = df_full_train[FEATURES]
y = df_full_train[TARGET].values
print("Training rows:", len(X), "| share of good wines:", round(y.mean(), 3))

# The same 5 folds are used for every model, so the scores are comparable.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Missing values are filled INSIDE the pipeline with the median of each training fold.
# (No information from the validation fold leaks into the fill values.)
impute = SimpleImputer(strategy="median")

models = {
    "dummy (always majority class)": DummyClassifier(strategy="prior"),
    "logistic regression": make_pipeline(impute, StandardScaler(), LogisticRegression(max_iter=1000)),
    "random forest": make_pipeline(impute, RandomForestClassifier(random_state=SEED, n_jobs=2)),
    "xgboost": make_pipeline(impute, XGBClassifier(random_state=SEED, n_jobs=2, eval_metric="logloss")),
}

# ---------- 1) Default parameters ----------
print("\n1) Cross-validated ROC AUC with default parameters:")
rows = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    rows.append({"model": name, "stage": "default", "auc_mean": scores.mean(), "auc_std": scores.std()})
    print(f"   {name:30s} AUC = {scores.mean():.4f} +- {scores.std():.4f}")

# ---------- 2) Hyperparameter tuning (randomized search) ----------
searches = {
    "logistic regression": (
        make_pipeline(impute, StandardScaler(), LogisticRegression(max_iter=2000)),
        {"logisticregression__C": loguniform(1e-3, 1e2)},
        15,
    ),
    "random forest": (
        make_pipeline(impute, RandomForestClassifier(random_state=SEED, n_jobs=2)),
        {
            "randomforestclassifier__n_estimators": [100, 200, 300, 500],
            "randomforestclassifier__max_depth": [None, 8, 12, 16, 20],
            "randomforestclassifier__min_samples_leaf": [1, 2, 3, 5],
            "randomforestclassifier__max_features": ["sqrt", 0.5, 0.7],
        },
        20,
    ),
    "xgboost": (
        make_pipeline(impute, XGBClassifier(random_state=SEED, n_jobs=2, eval_metric="logloss")),
        {
            "xgbclassifier__n_estimators": [100, 200, 300, 500],
            "xgbclassifier__max_depth": [3, 4, 5, 6, 8],
            "xgbclassifier__learning_rate": [0.02, 0.05, 0.1, 0.2],
            "xgbclassifier__min_child_weight": [1, 3, 5],
            "xgbclassifier__subsample": [0.7, 0.85, 1.0],
            "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
        },
        30,
    ),
}

print("\n2) Randomized search (5-fold CV, ROC AUC):")
best_params = {}
for name, (pipe, grid, n_iter) in searches.items():
    t0 = time.time()
    search = RandomizedSearchCV(
        pipe, grid, n_iter=n_iter, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1
    )
    search.fit(X, y)
    idx = search.best_index_
    std = search.cv_results_["std_test_score"][idx]
    rows.append({"model": name, "stage": "tuned", "auc_mean": search.best_score_, "auc_std": std})
    best_params[name] = {
        k.split("__")[1]: (v.item() if hasattr(v, "item") else v)  # numpy scalar -> plain Python
        for k, v in search.best_params_.items()
    }
    print(f"   {name:22s} best AUC = {search.best_score_:.4f} +- {std:.4f}  ({time.time() - t0:.0f}s)")
    print("      best params:", best_params[name])

# ---------- 3) Summary table ----------
res = pd.DataFrame(rows).round(4)
res.to_csv("results/model_comparison.csv", index=False)
print("\n3) Summary:")
print(res.pivot(index="model", columns="stage", values="auc_mean").round(4).to_string())

with open("results/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)


