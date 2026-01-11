import pandas as pd
import matplotlib.pyplot as plt

# --- Hard-coded values (edit these if needed) ---
CSV_PATH = "D:\PG\Pattern\E22WHnew.csv"
BINS = 25
SAVE_FIG = False
OUT_PATH = "D:\PG\Pattern\weight_height_histograms.png"
# ----------------------------------------------

# Read CSV (assumes NO header)
df = pd.read_csv(CSV_PATH, header=None)

# Columns:
# 0 = weight, 1 = height, 2 = sex (male=1, female=-1)
df = df.iloc[:, :3].copy()
df.columns = ["weight", "height", "sex"]

# Make sure everything is numeric and clean
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df["height"] = pd.to_numeric(df["height"], errors="coerce")
df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
df = df.dropna(subset=["weight", "height", "sex"])

male = df[df["sex"] == 1]
female = df[df["sex"] == -1]

# Plot: 2 histograms (Weight, Height) with male/female overlaid
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Weight
axes[0].hist(male["weight"], bins=BINS, alpha=0.6, label="Male (1)")
axes[0].hist(female["weight"], bins=BINS, alpha=0.6, label="Female (-1)")
axes[0].set_title("Weight Histogram")
axes[0].set_xlabel("Weight")
axes[0].set_ylabel("Count")
axes[0].legend()

# Height
axes[1].hist(male["height"], bins=BINS, alpha=0.6, label="Male (1)")
axes[1].hist(female["height"], bins=BINS, alpha=0.6, label="Female (-1)")
axes[1].set_title("Height Histogram")
axes[1].set_xlabel("Height")
axes[1].set_ylabel("Count")
axes[1].legend()

if SAVE_FIG:
    fig.savefig(OUT_PATH, dpi=200)
    print("Saved:", OUT_PATH)

plt.show()
