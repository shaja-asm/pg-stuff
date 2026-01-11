"""
Bayesian Decision Theory classifier using YOUR assumption:

  P(F|w,h) = P(F|h) * P(F|w)
  P(M|w,h) = P(M|h) * P(M|w)

We compute P(F|x) and P(M|x) (x is weight or height) via Bayes rule:
  P(F|x) = p(x|F)P(F) / (p(x|F)P(F) + p(x|M)P(M))

Likelihoods are Gaussian (as you gave):
  p(x|C) = 1/(sqrt(2π) σ_C) * exp(-0.5 * ((x-μ_C)/σ_C)^2)

Outputs (ALL THREE):
  1) Scatter plot with separating curve (male=red, female=blue)
  2) Test accuracy + confusion matrix
  3) ROC curve + AUC using the Bayes decision score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


CSV_PATH = r"D:\PG\Pattern\E22WHnew.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.30
EPS = 1e-12  # numerical stability


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns (weight, height, sex), got {df.shape[1]}.")

    df = df.iloc[:, :3].copy()
    df.columns = ["weight", "height", "sex"]

    for col in ("weight", "height", "sex"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["weight", "height", "sex"]).reset_index(drop=True)
    df = df[df["sex"].isin([1, -1])].reset_index(drop=True)

    if not ((df["sex"] == 1).any() and (df["sex"] == -1).any()):
        raise ValueError("Need both classes present: male=1 and female=-1.")

    return df


def fit_gaussian_params(x: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    sigma = max(sigma, 1e-9)
    return mu, sigma


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z * z)


def posterior_F_given_x(
    x: np.ndarray,
    mu_F: float,
    sig_F: float,
    mu_M: float,
    sig_M: float,
    prior_F: float,
    prior_M: float,
) -> np.ndarray:
    """P(F|x) using Bayes rule."""
    px_F = gaussian_pdf(x, mu_F, sig_F)
    px_M = gaussian_pdf(x, mu_M, sig_M)
    num = px_F * prior_F
    den = (px_F * prior_F) + (px_M * prior_M)
    return num / (den + EPS)


def fit_model(train_df: pd.DataFrame) -> dict:
    """Fit priors and Gaussian params from TRAIN split."""
    female = train_df[train_df["sex"] == -1]
    male = train_df[train_df["sex"] == 1]

    prior_F = len(female) / len(train_df)
    prior_M = len(male) / len(train_df)

    mu_Fw, sig_Fw = fit_gaussian_params(female["weight"].to_numpy(float))
    mu_Mw, sig_Mw = fit_gaussian_params(male["weight"].to_numpy(float))
    mu_Fh, sig_Fh = fit_gaussian_params(female["height"].to_numpy(float))
    mu_Mh, sig_Mh = fit_gaussian_params(male["height"].to_numpy(float))

    return {
        "prior_F": prior_F,
        "prior_M": prior_M,
        "mu_Fw": mu_Fw,
        "sig_Fw": sig_Fw,
        "mu_Mw": mu_Mw,
        "sig_Mw": sig_Mw,
        "mu_Fh": mu_Fh,
        "sig_Fh": sig_Fh,
        "mu_Mh": mu_Mh,
        "sig_Mh": sig_Mh,
    }


def decision_score(W: np.ndarray, H: np.ndarray, model: dict) -> np.ndarray:
    """
    D(w,h) = log P(M|w,h) - log P(F|w,h)

    Using YOUR assumption:
      P(F|w,h) = P(F|w)*P(F|h)
      P(M|w,h) = P(M|w)*P(M|h)

    Boundary: D(w,h) = 0
    """
    prior_F = model["prior_F"]
    prior_M = model["prior_M"]

    PF_w = posterior_F_given_x(W, model["mu_Fw"], model["sig_Fw"], model["mu_Mw"], model["sig_Mw"], prior_F, prior_M)
    PF_h = posterior_F_given_x(H, model["mu_Fh"], model["sig_Fh"], model["mu_Mh"], model["sig_Mh"], prior_F, prior_M)

    PM_w = 1.0 - PF_w
    PM_h = 1.0 - PF_h

    log_F = np.log(PF_w + EPS) + np.log(PF_h + EPS)
    log_M = np.log(PM_w + EPS) + np.log(PM_h + EPS)

    return log_M - log_F


def predict_labels(df: pd.DataFrame, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred in {1,-1}, scores)."""
    W = df["weight"].to_numpy(float)
    H = df["height"].to_numpy(float)
    scores = decision_score(W, H, model)
    y_pred = np.where(scores >= 0.0, 1, -1)  # Male if score >= 0
    return y_pred, scores


def plot_scatter_with_boundary(df: pd.DataFrame, model: dict) -> None:
    male = df[df["sex"] == 1]
    female = df[df["sex"] == -1]

    w_min, w_max = df["weight"].min(), df["weight"].max()
    h_min, h_max = df["height"].min(), df["height"].max()

    w_pad = 0.05 * (w_max - w_min + 1e-9)
    h_pad = 0.05 * (h_max - h_min + 1e-9)

    ww = np.linspace(w_min - w_pad, w_max + w_pad, 400)
    hh = np.linspace(h_min - h_pad, h_max + h_pad, 400)
    Wg, Hg = np.meshgrid(ww, hh)

    D = decision_score(Wg, Hg, model)

    plt.figure(figsize=(8, 6))
    plt.scatter(male["weight"], male["height"], c="red", alpha=0.7, label="Male (1)")
    plt.scatter(female["weight"], female["height"], c="blue", alpha=0.7, label="Female (-1)")

    # Separating curve where D(w,h)=0
    plt.contour(Wg, Hg, D, levels=[0.0], linewidths=2)

    plt.title("Scatter Plot with Bayes Separating Curve (Male=Red, Female=Blue)")
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc(y_true: np.ndarray, scores: np.ndarray) -> None:
    """
    ROC uses scores as a ranking signal.
    Positive class = Male (1).
    """
    y_bin = (y_true == 1).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, scores)
    roc_auc = auc(fpr, tpr)

    # If score direction is flipped, fix it
    if roc_auc < 0.5:
        fpr, tpr, _ = roc_curve(y_bin, -scores)
        roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curve (Bayesian Decision Score)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"AUC: {roc_auc:.4f}")


def main() -> None:
    df = load_data(CSV_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sex"],
    )

    model = fit_model(train_df)

    # (1) Scatter + separating curve (on full dataset for visualization; boundary from train-fit model)
    plot_scatter_with_boundary(df, model)

    # (2) Accuracy + confusion matrix (on test set)
    y_true = test_df["sex"].to_numpy(int)
    y_pred, scores = predict_labels(test_df, model)

    acc = float(np.mean(y_pred == y_true))
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])  # rows=true, cols=pred
    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix (labels [Male=1, Female=-1]):")
    print(cm)

    # (3) ROC + AUC (on test set)
    plot_roc(y_true, scores)


if __name__ == "__main__":
    main()
