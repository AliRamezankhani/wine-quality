"""Train the final model and save it to a file.

Run:  python train.py

Steps:
1. load and clean the data, split it into train and test (same split as everywhere else)
2. train the final Random Forest (parameters found in model_selection.py) on the whole training part
3. evaluate ONCE on the untouched test set
4. save the model pipeline to model/model.bin and the metrics to results/test_metrics.json
"""
import json
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from data_prep import FEATURES, TARGET, load_and_clean, split_data

SEED = 1
MODEL_PATH = "model/model.bin"

# Best parameters from model_selection.py (5-fold CV, ROC AUC)
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
}


def train(df_train):
    """Fit the pipeline: median imputation + Random Forest."""
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),  # medians are learned from training data only
        RandomForestClassifier(**BEST_PARAMS, random_state=SEED, n_jobs=2),
    )
    pipeline.fit(df_train[FEATURES], df_train[TARGET].values)
    return pipeline


def evaluate(pipeline, df_eval):
    """Return ROC AUC and accuracy (threshold 0.5) on the given data."""
    y_true = df_eval[TARGET].values
    y_proba = pipeline.predict_proba(df_eval[FEATURES])[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_proba >= 0.5)), 4),
        "n_rows": int(len(df_eval)),
    }


if __name__ == "__main__":
    df = load_and_clean()
    df_full_train, df_test = split_data(df)
    print(f"Train rows: {len(df_full_train)} | test rows: {len(df_test)}")

    print("Training the final model...")
    model = train(df_full_train)

    train_metrics = evaluate(model, df_full_train)
    test_metrics = evaluate(model, df_test)
    print("Train metrics:", train_metrics, "(a Random Forest fits its own training data very well)")
    print("Test metrics: ", test_metrics, "<- the honest estimate for new wines")

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump(model, f_out)
    print(f"Model saved to {MODEL_PATH}")

    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics.json", "w") as f_out:
        json.dump({"train": train_metrics, "test": test_metrics, "params": BEST_PARAMS}, f_out, indent=2)