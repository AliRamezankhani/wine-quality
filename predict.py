"""Web service (Flask) that serves the trained wine quality model.

Run locally:  python predict.py      (listens on http://localhost:9696)
"""
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from data_prep import FEATURES

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.bin"
THRESHOLD = 0.5  # probability >= 0.5 -> "good"

# The 11 lab measurements the client must send (is_red is built from "type")
LAB_FIELDS = [f for f in FEATURES if f != "is_red"]

with open(MODEL_PATH, "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("wine-quality")


def parse_wine(data):
    """Validate one JSON wine and turn it into a one-row DataFrame in the model's feature order."""
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")

    missing = [f for f in LAB_FIELDS + ["type"] if f not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    wine_type = str(data["type"]).lower()
    if wine_type not in ("red", "white"):
        raise ValueError("`type` must be 'red' or 'white'")

    row = {}
    for field in LAB_FIELDS:
        value = data[field]
        if value is None:
            row[field] = float("nan")  # the model fills missing values with training medians
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            row[field] = float(value)
        else:
            raise ValueError(f"`{field}` must be a number")
    row["is_red"] = 1 if wine_type == "red" else 0

    return pd.DataFrame([row], columns=FEATURES)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    try:
        X = parse_wine(data)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    probability = float(model.predict_proba(X)[0, 1])
    return jsonify({
        "good_probability": round(probability, 4),
        "good": bool(probability >= THRESHOLD),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=9696)