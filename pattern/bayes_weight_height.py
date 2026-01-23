"""
Bayesian Decision Theory classifier using the assumptions given in class

(1) 1D Bayes posteriors:
    P(F|x) = α_x * p(x|F) P(F)
    P(M|x) = α_x * p(x|M) P(M)
    where α_x = 1 / (p(x|F)P(F) + p(x|M)P(M))
    (x is either h or w)

(2) GIVEN combination assumption:
    P(F|w,h) = P(F|h) * P(F|w)
    P(M|w,h) = P(M|h) * P(M|w)

    These products are NOT guaranteed to sum to 1, so we renormalise with β(w,h):
      F_unn = P(F|h)P(F|w)
      M_unn = P(M|h)P(M|w)
      β = 1 / (F_unn + M_unn)

      P(F|w,h) = β * F_unn
      P(M|w,h) = β * M_unn

Decision rule:
    Male (1) if P(M|w,h) >= 0.5 else Female (-1)

Outputs:
  (1) Scatter plot + decision boundary contour P(M|w,h)=0.5
  (2) Test accuracy + balanced accuracy + confusion matrix + classification report
  (3) ROC + AUC using P(M|w,h) as the score
  (4) Precision–Recall curve + Average Precision
  (5) Prints fitted Gaussian parameters + priors

Labels:
  Male = 1, Female = -1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    balanced_accuracy_score,
    precision_recall_curve,
    average_precision_score,
)


CSV_PATH = r"D:\PG\Pattern\E22WHnew.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.30
EPS = 1e-12


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

# GAUSSIAN HELPERS
def fit_gaussian_params(x: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    sigma = max(sigma, 1e-9)
    return mu, sigma


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z * z)

# MODEL FIT
def fit_model(train_df: pd.DataFrame) -> dict:
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
        "mu_Fw": mu_Fw, "sig_Fw": sig_Fw,
        "mu_Mw": mu_Mw, "sig_Mw": sig_Mw,
        "mu_Fh": mu_Fh, "sig_Fh": sig_Fh,
        "mu_Mh": mu_Mh, "sig_Mh": sig_Mh,
    }


def print_model_params(model: dict) -> None:
    print("\n=== Fitted parameters (TRAIN split) ===")
    print(f"Priors: P(M)={model['prior_M']:.4f}, P(F)={model['prior_F']:.4f}")
    print(
        f"Weight: mu_M={model['mu_Mw']:.3f}, sig_M={model['sig_Mw']:.3f} | "
        f"mu_F={model['mu_Fw']:.3f}, sig_F={model['sig_Fw']:.3f}"
    )
    print(
        f"Height: mu_M={model['mu_Mh']:.3f}, sig_M={model['sig_Mh']:.3f} | "
        f"mu_F={model['mu_Fh']:.3f}, sig_F={model['sig_Fh']:.3f}"
    )
    print("=======================================\n")

# 1D BAYES POSTERIORS (α_h and α_w happen inside the formula)
def posterior_F_given_x(
    x: np.ndarray,
    mu_F: float,
    sig_F: float,
    mu_M: float,
    sig_M: float,
    prior_F: float,
    prior_M: float,
) -> np.ndarray:
    """
    P(F|x) = [p(x|F)P(F)] / [p(x|F)P(F) + p(x|M)P(M)]
    The denominator is exactly 1/α_x.
    """
    px_F = gaussian_pdf(x, mu_F, sig_F)
    px_M = gaussian_pdf(x, mu_M, sig_M)
    num = px_F * prior_F
    den = (px_F * prior_F) + (px_M * prior_M)
    return num / (den + EPS)

# 2D COMBINATION + β NORMALISATION
def posterior_M_given_wh(W: np.ndarray, H: np.ndarray, model: dict) -> np.ndarray:
    """
    Implements your GIVEN assumption + correct β normalisation.

    1) Compute 1D posteriors:
       PF_w = P(F|w), PF_h = P(F|h)
       PM_w = 1 - PF_w, PM_h = 1 - PF_h

    2) Combine (as given):
       F_unn = PF_w * PF_h
       M_unn = PM_w * PM_h

    3) Normalise using β(w,h):
       β = 1 / (F_unn + M_unn)
       PM_wh = β * M_unn
    """
    prior_F = model["prior_F"]
    prior_M = model["prior_M"]

    PF_w = posterior_F_given_x(
        W, model["mu_Fw"], model["sig_Fw"], model["mu_Mw"], model["sig_Mw"], prior_F, prior_M
    )
    PF_h = posterior_F_given_x(
        H, model["mu_Fh"], model["sig_Fh"], model["mu_Mh"], model["sig_Mh"], prior_F, prior_M
    )

    PM_w = 1.0 - PF_w
    PM_h = 1.0 - PF_h

    F_unn = PF_w * PF_h
    M_unn = PM_w * PM_h

    beta = 1.0 / (F_unn + M_unn + EPS)
    PM_wh = beta * M_unn
    return PM_wh


def predict_labels(df: pd.DataFrame, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_pred in {1,-1}
      score_pm = P(M|w,h) in [0,1] for ROC/PR
    """
    W = df["weight"].to_numpy(float)
    H = df["height"].to_numpy(float)

    score_pm = posterior_M_given_wh(W, H, model)
    y_pred = np.where(score_pm >= 0.5, 1, -1)
    return y_pred, score_pm

# PLOTS
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

    PM_grid = posterior_M_given_wh(Wg, Hg, model)

    plt.figure(figsize=(8, 6))
    plt.scatter(male["weight"], male["height"], c="red", alpha=0.7, label="Male (1)")
    plt.scatter(female["weight"], female["height"], c="blue", alpha=0.7, label="Female (-1)")

    # Boundary: P(M|w,h) = 0.5
    plt.contour(Wg, Hg, PM_grid, levels=[0.5], linewidths=2)

    plt.title("Scatter Plot with Bayes Separating Curve (P(M|w,h)=0.5)")
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc(y_true: np.ndarray, score_pm: np.ndarray) -> float:
    y_bin = (y_true == 1).astype(int)  # positive = Male
    fpr, tpr, _ = roc_curve(y_bin, score_pm)
    roc_auc = auc(fpr, tpr)

    if roc_auc < 0.5:
        fpr, tpr, _ = roc_curve(y_bin, 1.0 - score_pm)
        roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curve (Posterior Score P(M|w,h))")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()
    return roc_auc


def plot_precision_recall(y_true: np.ndarray, score_pm: np.ndarray) -> float:
    y_bin = (y_true == 1).astype(int)
    prec, rec, _ = precision_recall_curve(y_bin, score_pm)
    ap = average_precision_score(y_bin, score_pm)

    plt.figure(figsize=(6, 6))
    plt.plot(rec, prec, label=f"PR (AP = {ap:.3f})")
    plt.title("Precision–Recall Curve (Posterior Score P(M|w,h))")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.show()
    return ap

def main() -> None:
    df = load_data(CSV_PATH)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sex"],
    )

    model = fit_model(train_df)
    print_model_params(model)

    # (1) Scatter + separating curve
    plot_scatter_with_boundary(df, model)

    # (2) Metrics on test set
    y_true = test_df["sex"].to_numpy(int)
    y_pred, score_pm = predict_labels(test_df, model)

    acc = float(np.mean(y_pred == y_true))
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])

    print(f"Test accuracy: {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print("Confusion matrix (labels [Male=1, Female=-1]):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[1, -1], target_names=["Male (1)", "Female (-1)"]))

    # (3) ROC + AUC
    roc_auc = plot_roc(y_true, score_pm)
    print(f"AUC: {roc_auc:.4f}")

    # (4) PR + AP
    ap = plot_precision_recall(y_true, score_pm)
    print(f"Average Precision (AP): {ap:.4f}")


if __name__ == "__main__":
    main()
