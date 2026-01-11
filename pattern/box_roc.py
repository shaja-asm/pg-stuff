"""
Plots:
  1) Box plot (Weight): Male vs Female
  2) Box plot (Height): Male vs Female
  3) ROC curve using ONLY Weight
  4) ROC curve using ONLY Height

CSV format (no header expected):
  col 0 = weight
  col 1 = height
  col 2 = sex  (male=1, female=-1)
"""
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CSV_PATH = r"D:\PG\Pattern\E22WHnew.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.30


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the dataset, returning columns: weight, height, sex."""
    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns (weight, height, sex), got {df.shape[1]}.")

    df = df.iloc[:, :3].copy()
    df.columns = ["weight", "height", "sex"]

    for col in ("weight", "height", "sex"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["weight", "height", "sex"]).reset_index(drop=True)
    return df


def split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return male and female dataframes."""
    male = df[df["sex"] == 1]
    female = df[df["sex"] == -1]

    if male.empty or female.empty:
        raise ValueError(
            f"One of the groups is empty. male rows={len(male)}, female rows={len(female)}. "
            "Verify sex column uses male=1 and female=-1."
        )

    return male, female


def plot_boxplot_weight(male: pd.DataFrame, female: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 5))
    plt.boxplot([male["weight"], female["weight"]], labels=["Male (1)", "Female (-1)"])
    plt.title("Weight Box Plot: Male vs Female")
    plt.ylabel("Weight")
    plt.grid(True, axis="y")
    plt.show()


def plot_boxplot_height(male: pd.DataFrame, female: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 5))
    plt.boxplot([male["height"], female["height"]], labels=["Male (1)", "Female (-1)"])
    plt.title("Height Box Plot: Male vs Female")
    plt.ylabel("Height")
    plt.grid(True, axis="y")
    plt.show()


def plot_single_feature_roc(df: pd.DataFrame, feature: str, title: str) -> None:
    """
    Plot ROC curve using ONLY one feature ('weight' or 'height') to predict Male(1) vs Female(-1).
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe columns: {list(df.columns)}")

    X = df[[feature]].to_numpy()
    y = (df["sex"] == 1).astype(int).to_numpy()  # male=1, female=0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]  # P(male)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    df = load_data(CSV_PATH)
    male, female = split_groups(df)

    # Box plots
    plot_boxplot_weight(male, female)
    plot_boxplot_height(male, female)

    # ROC curves separately
    plot_single_feature_roc(df, feature="weight", title="ROC Curve (Weight Only): Male vs Female")
    plot_single_feature_roc(df, feature="height", title="ROC Curve (Height Only): Male vs Female")


if __name__ == "__main__":
    main()
