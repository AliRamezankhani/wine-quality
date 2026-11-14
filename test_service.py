"""Send a few real wines from the test set to the running web service and compare with the truth.

1. Start the service (Docker or `python predict.py`)
2. Run:  python test_service.py            (uses http://localhost:9696)
"""
import json
import sys
import urllib.request

from data_prep import FEATURES, TARGET, load_and_clean, split_data

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696"
N_WINES = 8


def post_json(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


with urllib.request.urlopen(f"{URL}/health", timeout=10) as response:
    print("health:", json.loads(response.read()))

_, df_test = split_data(load_and_clean())
sample = df_test.sample(N_WINES, random_state=1)

correct = 0
for _, row in sample.iterrows():
    wine = {f: (None if row[f] != row[f] else float(row[f])) for f in FEATURES if f != "is_red"}  # NaN -> null
    wine["type"] = "red" if row["is_red"] == 1 else "white"
    result = post_json(f"{URL}/predict", wine)
    truth = bool(row[TARGET])
    correct += result["good"] == truth
    print(f"{wine['type']:5s} alcohol={wine['alcohol']:5.1f} -> P(good)={result['good_probability']:.3f} "
          f"predicted={'good' if result['good'] else 'not good':8s} real={'good' if truth else 'not good'}")

print(f"\nCorrect: {correct} of {N_WINES}")
