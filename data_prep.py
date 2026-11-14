"""Shared data loading and cleaning, used by the notebook, EDA script and train.py."""
from pathlib import Path

import pandas as pd

DATA_FILE = "wine_quality.csv"


def _find_data_path() -> Path:
    """Look for data/<DATA_FILE> in the folder of this file and in all parent folders."""
    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        candidate = folder / "data" / DATA_FILE
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find data/{DATA_FILE} next to or above {here}")


DATA_PATH = _find_data_path()
GOOD_THRESHOLD = 6  # a wine with quality >= 6 is labelled "good"


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a cleaned DataFrame with a binary target `good`.

    Missing values are NOT filled here on purpose: the fill values (medians) must be
    learned on the training data only, so imputation happens later inside the model pipeline.
    """
    df = pd.read_csv(path)

    # Consistent column names: lowercase, underscores instead of spaces
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # The raw file contains exact duplicate wines. If the same wine lands in both
    # train and validation, scores look better than they really are (data leakage).
    df = df.drop_duplicates().reset_index(drop=True)

    # Wine type (red / white) -> numeric 0/1 feature
    df["is_red"] = (df["type"].str.lower() == "red").astype(int)
    df = df.drop(columns=["type"])

    # Binary target: 1 = good wine (quality >= 6), 0 = otherwise
    df["good"] = (df["quality"] >= GOOD_THRESHOLD).astype(int)

    # `quality` directly defines the target, so it must never be used as a feature
    return df


FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol", "is_red",
]
TARGET = "good"


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 1):
    """Split into (full_train, test). Stratified, so good/not-good shares are the same in both.

    Every script uses this one function, so everybody gets exactly the same split.
    """
    from sklearn.model_selection import train_test_split

    df_full_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET]
    )
    return df_full_train.reset_index(drop=True), df_test.reset_index(drop=True)