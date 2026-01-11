"""
Fisher’s Discriminant Analysis (2-class LDA) on Weight/Height data
+ ROC curve computed from Fisher projection scores.

CSV format (no header expected):
  col 0 = weight
  col 1 = height
  col 2 = sex  (male=1, female=-1)

What this script does:
  1) Splits data into train/test
  2) Fits Fisher direction w on train
  3) Uses Fisher scores s(x)=w^T x on test as the “classifier score”
  4) Plots ROC curve + prints AUC
  5) (Optional) prints confusion matrix for the midpoint threshold on train projections
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


CSV_PATH = r"D:\PG\Pattern\E22WHnew.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.30


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return X (n,2) and y (n,) with y in {1,-1}."""
    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns (weight, height, sex), got {df.shape[1]}.")

    df = df.iloc[:, :3].copy()
    df.columns = ["weight", "height", "sex"]

    for col in ("weight", "height", "sex"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["weight", "height", "sex"]).reset_index(drop=True)

    X = df[["weight", "height"]].to_numpy(dtype=float)
    y = df["sex"].to_numpy(dtype=int)

    mask = np.isin(y, [1, -1])
    X, y = X[mask], y[mask]

    if not (np.any(y == 1) and np.any(y == -1)):
        raise ValueError("Both classes are required: male=1 and female=-1.")

    return X, y


def fisher_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Fit Fisher direction w and choose a simple threshold t (midpoint of projected class means).
    Decision rule: predict +1 if w^T x >= t else -1
    """
    X1 = X[y == 1]
    X2 = X[y == -1]

    m1 = X1.mean(axis=0)
    m2 = X2.mean(axis=0)

    S1 = (X1 - m1).T @ (X1 - m1)
    S2 = (X2 - m2).T @ (X2 - m2)
    Sw = S1 + S2

    w = np.linalg.pinv(Sw) @ (m1 - m2)

    w_norm = np.linalg.norm(w)
    if w_norm > 0:
        w = w / w_norm

    mu1 = float(m1 @ w)
    mu2 = float(m2 @ w)
    t = 0.5 * (mu1 + mu2)

    return w, t


def fisher_scores(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Projection scores s(x)=w^T x."""
    return X @ w


def predict_from_threshold(scores: np.ndarray, t: float) -> np.ndarray:
    """Predict labels in {1,-1} from 1D scores and threshold."""
    return np.where(scores >= t, 1, -1)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion matrix dict assuming positive class is +1."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def plot_fisher_roc(y_test: np.ndarray, scores_test: np.ndarray) -> None:
    """
    Plot ROC using Fisher scores as the ranking signal.
    sklearn expects y in {0,1}; we map male(+1)->1, female(-1)->0.
    """
    y_bin = (y_test == 1).astype(int)

    fpr, tpr, thresholds = roc_curve(y_bin, scores_test)
    roc_auc = auc(fpr, tpr)

    # If direction got flipped (rare), AUC can come out < 0.5; fix by flipping scores
    if roc_auc < 0.5:
        fpr, tpr, thresholds = roc_curve(y_bin, -scores_test)
        roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Fisher ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curve from Fisher Discriminant Scores")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"AUC (Fisher scores on test set): {roc_auc:.4f}")


def main() -> None:
    X, y = load_data(CSV_PATH)

    # Split for a more honest ROC estimate
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    w, t = fisher_fit(X_train, y_train)

    # Scores for ROC (use test scores!)
    scores_test = fisher_scores(X_test, w)
    plot_fisher_roc(y_test, scores_test)

    # Optional: show threshold-based classification metrics (on test, using same t)
    y_pred_test = predict_from_threshold(scores_test, t)
    cm = confusion_matrix_binary(y_test, y_pred_test)
    acc = float(np.mean(y_pred_test == y_test))

    print("Fisher direction w (unit-norm):", w)
    print("Threshold t (midpoint on train projections):", t)
    print("Confusion Matrix on test (Male=+1):", cm)
    print(f"Accuracy on test (using threshold t): {acc:.4f}")


if __name__ == "__main__":
    main()
