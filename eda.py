# %%
import os
from pathlib import Path 
import matplotlib
matplotlib.use("Agg")  # no display needed; remove this line in Jupyter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

from data_prep import load_and_clean, FEATURES, TARGET

PROJECT_ROOT = Path.cwd()
IMAGE_PATH = PROJECT_ROOT / "images"

df = load_and_clean()

print(df.shape)

# %%
# ---------- Split Data ----------
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1, stratify=df[TARGET]
)
df_full_train = df_full_train.reset_index(drop=True)
print("\n2) full_train:", df_full_train.shape, "| test:", df_test.shape)

# %%
train_medians = df_full_train[FEATURES].median()
df_full_train[FEATURES] = df_full_train[FEATURES].fillna(train_medians)

# %%
#  ---------- Target analysis ----------
print("\n3) Quality score distribution (full_train):")
print(df_full_train.quality.value_counts().sort_index().to_string())
print("   Share of good wines (quality >= 6):", round(df_full_train[TARGET].mean(), 3))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
df_full_train.quality.value_counts().sort_index().plot.bar(ax=ax[0], color="tab:purple")
ax[0].set_title("Quality score distribution")
ax[0].set_xlabel("quality")
df_full_train[TARGET].value_counts().sort_index().plot.bar(ax=ax[1], color="tab:blue")
ax[1].set_title("Target: good (1) vs not good (0)")
ax[1].set_xlabel("good")
plt.tight_layout()
plt.savefig(IMAGE_PATH / "target_distribution.png", dpi=110)
plt.close(fig)

# %%
# ---------- 4) Feature ranges and outliers ----------
print("\n4) Feature summary (full_train):")
print(df_full_train[FEATURES].describe().T[["min", "mean", "max"]].round(3).to_string())
 
print("\n   Outliers by the 1.5*IQR rule:")
for c in [f for f in FEATURES if f != "is_red"]:
    q1, q3 = df_full_train[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = int(((df_full_train[c] < q1 - 1.5 * iqr) | (df_full_train[c] > q3 + 1.5 * iqr)).sum())
    print(f"   {c:22s} {n_out:4d}")
 
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for ax, c in zip(axes.ravel(), FEATURES):
    ax.hist(df_full_train[c], bins=30, color="tab:gray")
    ax.set_title(c, fontsize=9)
plt.tight_layout()
plt.savefig(IMAGE_PATH / "feature_histograms.png", dpi=110)
plt.close(fig)

# %%
# ---------- Feature importance: correlation with the target (numeric features) ----------
corr = df_full_train[FEATURES].corrwith(df_full_train[TARGET]).sort_values()
print("\n5) Correlation with `good`:")
print(corr.round(3).to_string())

# %%
# ---------- Feature importance: mutual information (also catches non-linear links) ----------
mi = pd.Series(
    mutual_info_classif(df_full_train[FEATURES], df_full_train[TARGET], random_state=1),
    index=FEATURES,
).sort_values(ascending=False)
print("\n6) Mutual information with `good`:")
print(mi.round(3).to_string())
 
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
corr.plot.barh(ax=ax[0], color=np.where(corr > 0, "tab:green", "tab:red"))
ax[0].set_title("Correlation with good")
mi.sort_values().plot.barh(ax=ax[1], color="tab:blue")
ax[1].set_title("Mutual information with good")
plt.tight_layout()
plt.savefig(IMAGE_PATH / "feature_importance_eda.png", dpi=110)
plt.close(fig)

# %%
# ---------- Correlation between features ----------
cm = df_full_train[FEATURES].corr()
pairs = (
    cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    .stack().rename("corr").to_frame()
)
pairs["abs"] = pairs["corr"].abs()
print("\n7) Strongest feature-feature correlations:")
print(pairs.sort_values("abs", ascending=False).head(5)["corr"].round(3).to_string())
 
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.savefig(IMAGE_PATH / "correlation_heatmap.png", dpi=110)
plt.close(fig)

# %%
# ---------- Share of good wines by wine type ----------
print("\n8) Share of good wines by type (is_red: 1 = red, 0 = white):")
print(df_full_train.groupby("is_red")[TARGET].agg(["count", "mean"]).round(3).to_string())

# %%
# ---------- Good vs not-good: average value of the top features ----------
print("\n9) Mean value by group:")
print(df_full_train.groupby(TARGET)[["alcohol", "volatile_acidity", "sulphates", "citric_acid"]]
      .mean().round(3).to_string())
 
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, c in zip(axes, ["alcohol", "volatile_acidity", "sulphates", "citric_acid"]):
    sns.boxplot(data=df_full_train, x=TARGET, y=c, ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.savefig(IMAGE_PATH / "top_features_boxplots.png", dpi=110)
plt.close(fig)


