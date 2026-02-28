#!/usr/bin/env python3
"""
CKDu_processed.csv multi-class analysis suite (8 analyses).

Dataset assumptions (per your description):
  - No header row
  - Col 0: class label in {1..5}
  - Col 1: Age
  - Col 2: SCR (Serum Creatinine)
  - Col 3..end: 30 unnamed features (F1..F30)

Analyses implemented:
  1) PCA-based analysis
  2) Bayesian Decision Theory (Gaussian Naive Bayes, multi-class)
  3) Fisher Discriminant analysis (LDA)
  4) Kurtosis vs skewness (shape diagnostics)
  5) Feature-wise statistical significance (-log10 p-value vs feature)
  6) Linear + Logistic regression
  7) Gaussian Mixture Models (one GMM per class, BIC model selection)
  8) Neural networks (MLP) + permutation importance

Outputs:
  - Saves figures + CSV tables into --outdir/<analysis>/
  - Prints key metrics + top-ranked features to console

Run examples:
  python ckdu_analysis_suite.py --analysis pca
  python ckdu_analysis_suite.py --analysis bayes
  python ckdu_analysis_suite.py --analysis fisher
  python ckdu_analysis_suite.py --analysis skew_kurt
  python ckdu_analysis_suite.py --analysis significance
  python ckdu_analysis_suite.py --analysis regression
  python ckdu_analysis_suite.py --analysis gmm
  python ckdu_analysis_suite.py --analysis nn

Tip:
  - If your CSV is elsewhere:  --csv /path/to/CKDu_processed.csv
  - If you want to exclude Age+SCR: --no_age_scr
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
    roc_auc_score,
)
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

LABEL_MAP = {
    1: "CKDu",
    2: "EC",
    3: "NEC",
    4: "ECKD",
    5: "NECKD",
}


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    class_names: List[str]


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def load_ckdu_processed(csv_path: str, use_age_scr: bool = True) -> Dataset:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"Expected >=4 columns, got {df.shape[1]}")

    y = df.iloc[:, 0].astype(int).to_numpy()
    mask = np.isin(y, list(LABEL_MAP.keys()))
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]

    age = df.iloc[:, 1].to_numpy(dtype=float)
    scr = df.iloc[:, 2].to_numpy(dtype=float)
    feats = df.iloc[:, 3:].to_numpy(dtype=float)

    if use_age_scr:
        X = np.column_stack([age, scr, feats])
        feature_names = ["Age", "SCR"] + [f"F{i}" for i in range(1, feats.shape[1] + 1)]
    else:
        X = feats
        feature_names = [f"F{i}" for i in range(1, feats.shape[1] + 1)]

    class_names = [LABEL_MAP[i] for i in sorted(np.unique(y))]
    return Dataset(X=X, y=y, feature_names=feature_names, class_names=class_names)


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _save_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    label_names: List[str],
    outpath: str,
    title: str,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# =====================
# 1) PCA-based analysis
# =====================
def analysis_pca(ds: Dataset, outdir: str, n_components: int = 10):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(n_components, X.shape[1]), random_state=RANDOM_STATE)),
    ])
    X_pca = pipe.fit_transform(X)
    pca: PCA = pipe.named_steps["pca"]

    # Explained variance plot
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(np.arange(1, len(evr) + 1), evr, marker="o", linewidth=1.5, label="Explained variance")
    ax.plot(np.arange(1, len(evr) + 1), cum, marker="s", linewidth=1.5, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_explained_variance.png"), dpi=180)
    plt.close(fig)

    # Scatter PC1 vs PC2
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    for lab in labels_sorted:
        idx = y == lab
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], s=28, alpha=0.8, label=LABEL_MAP.get(int(lab), str(lab)))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection: PC1 vs PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_scatter_pc1_pc2.png"), dpi=180)
    plt.close(fig)

    # Loadings (feature contributions to PCs)
    loadings = pca.components_.T
    load_df = pd.DataFrame(loadings, index=ds.feature_names,
                           columns=[f"PC{i}" for i in range(1, loadings.shape[1] + 1)])
    load_df.to_csv(os.path.join(outdir, "pca_loadings.csv"))

    # Heuristic "which features drive separation for CKDu":
    # class centroids in PC space -> direction from overall centroid to CKDu centroid -> back-project.
    ckdu_label = 1
    if ckdu_label in labels_sorted:
        centroids = {lab: X_pca[y == lab].mean(axis=0) for lab in labels_sorted}
        overall = X_pca.mean(axis=0)
        d = centroids[ckdu_label] - overall
        M = min(n_components, len(d))
        feat_scores = loadings[:, :M] @ d[:M]
        sep_df = pd.DataFrame({
            "feature": ds.feature_names,
            "pca_ckdu_separation_score": feat_scores,
            "abs_score": np.abs(feat_scores),
        }).sort_values("abs_score", ascending=False)
        sep_df.to_csv(os.path.join(outdir, "pca_ckdu_separation_feature_scores.csv"), index=False)
        print("\n[PCA] Top features driving CKDu centroid separation (|score|):")
        print(sep_df.head(12).to_string(index=False))
    else:
        print("[PCA] CKDu label 1 not found in y; skipping CKDu separation scoring.")

    print(f"\n[PCA] Saved plots/tables to: {outdir}")


# =================================
# 2) Bayesian Decision Theory (multi)
# =================================
def _log_gauss_pdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    eps = 1e-9
    var = np.maximum(var, eps)
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var


def analysis_bayes(ds: Dataset, outdir: str, n_splits: int = 5):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred_oof = np.zeros_like(y)
    y_proba_oof = np.zeros((len(y), len(labels_sorted)), dtype=float)

    # var_smoothing stabilizes per-class variances (also stabilizes LLR contributions).
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("nb", GaussianNB(var_smoothing=1e-3)),
    ])

    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        y_pred_oof[te] = pipe.predict(X[te])

        proba = pipe.predict_proba(X[te])
        classes = pipe.named_steps["nb"].classes_
        col_idx = [np.where(classes == l)[0][0] for l in labels_sorted]
        y_proba_oof[te, :] = proba[:, col_idx]

    metrics = _basic_metrics(y, y_pred_oof)
    try:
        metrics["auc_macro_ovr"] = float(roc_auc_score(y, y_proba_oof, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc_macro_ovr"] = float("nan")

    with open(os.path.join(outdir, "bayes_cv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _save_confusion(
        y_true=y,
        y_pred=y_pred_oof,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "bayes_confusion_matrix.png"),
        title=f"Gaussian Naive Bayes (OOF, {n_splits}-fold CV)",
    )

    print("\n[Bayes] CV metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n[Bayes] Classification report (OOF):")
    print(classification_report(y, y_pred_oof, target_names=label_names, digits=3))

    # Single-feature ranking (macro-F1) with CV
    single_rows = []
    for j, feat in enumerate(ds.feature_names):
        y_pred = np.zeros_like(y)
        for tr, te in skf.split(X[:, [j]], y):
            pipe.fit(X[tr][:, [j]], y[tr])
            y_pred[te] = pipe.predict(X[te][:, [j]])
        single_rows.append((feat, f1_score(y, y_pred, average="macro"), accuracy_score(y, y_pred)))
    single_df = pd.DataFrame(single_rows, columns=["feature", "cv_f1_macro", "cv_accuracy"]).sort_values(
        "cv_f1_macro", ascending=False
    )
    single_df.to_csv(os.path.join(outdir, "bayes_single_feature_ranking.csv"), index=False)
    print("\n[Bayes] Top single-feature predictors (by CV macro-F1):")
    print(single_df.head(12).to_string(index=False))

    # Leave-one-feature-out drop (macro-F1)
    base_f1 = metrics["f1_macro"]
    loo_rows = []
    for j, feat in enumerate(ds.feature_names):
        cols = [i for i in range(X.shape[1]) if i != j]
        y_pred = np.zeros_like(y)
        for tr, te in skf.split(X[:, cols], y):
            pipe.fit(X[tr][:, cols], y[tr])
            y_pred[te] = pipe.predict(X[te][:, cols])
        f1m = f1_score(y, y_pred, average="macro")
        loo_rows.append((feat, base_f1 - f1m))
    loo_df = pd.DataFrame(loo_rows, columns=["feature", "f1_drop_when_removed"]).sort_values(
        "f1_drop_when_removed", ascending=False
    )
    loo_df.to_csv(os.path.join(outdir, "bayes_leave_one_out_importance.csv"), index=False)
    print("\n[Bayes] Top features by macro-F1 drop when removed (bigger = more important):")
    print(loo_df.head(12).to_string(index=False))

    # Decision-rule contribution toward CKDu (1) vs each other class using per-feature LLRs
    pipe.fit(X, y)
    nb: GaussianNB = pipe.named_steps["nb"]
    scaler: StandardScaler = pipe.named_steps["scaler"]
    Xs = scaler.transform(X)
    mu = nb.theta_
    var = nb.var_
    classes = nb.classes_

    if 1 in classes:
        idx_ckdu = np.where(classes == 1)[0][0]
        other_idx = [i for i, c in enumerate(classes) if c != 1]

        llr_abs_accum = np.zeros(X.shape[1])
        llr_signed_ckdu_accum = np.zeros(X.shape[1])
        for k in other_idx:
            llr = _log_gauss_pdf(Xs, mu[idx_ckdu], var[idx_ckdu]) - _log_gauss_pdf(Xs, mu[k], var[k])
            llr_abs_accum += np.mean(np.abs(llr), axis=0)
            llr_signed_ckdu_accum += np.mean(llr[y == 1], axis=0)

        llr_df = pd.DataFrame({
            "feature": ds.feature_names,
            "avg_abs_llr_vs_each_class": llr_abs_accum / max(len(other_idx), 1),
            "avg_signed_llr_on_ckdu_samples": llr_signed_ckdu_accum / max(len(other_idx), 1),
        }).sort_values("avg_abs_llr_vs_each_class", ascending=False)
        llr_df.to_csv(os.path.join(outdir, "bayes_llr_feature_contributions.csv"), index=False)
        print("\n[Bayes] Top features by |LLR| contribution (CKDu vs each other class):")
        print(llr_df.head(12).to_string(index=False))

    print(f"\n[Bayes] Saved plots/tables to: {outdir}")


# =============================
# 3) Fisher Discriminant (LDA)
# =============================
def analysis_fisher(ds: Dataset, outdir: str, n_splits: int = 5):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred_oof = np.zeros_like(y)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        y_pred_oof[te] = pipe.predict(X[te])

    metrics = _basic_metrics(y, y_pred_oof)
    with open(os.path.join(outdir, "fisher_lda_cv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _save_confusion(
        y_true=y,
        y_pred=y_pred_oof,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "fisher_lda_confusion_matrix.png"),
        title=f"Fisher LDA (OOF, {n_splits}-fold CV)",
    )
    print("\n[Fisher/LDA] CV metrics:")
    print(json.dumps(metrics, indent=2))

    # Fit once on all data for interpretability (one-vs-rest decision functions)
    pipe.fit(X, y)
    lda: LinearDiscriminantAnalysis = pipe.named_steps["lda"]
    coef = getattr(lda, "coef_", None)
    classes = lda.classes_
    if coef is not None and 1 in classes:
        ckdu_idx = np.where(classes == 1)[0][0]
        ckdu_coef = coef[ckdu_idx]
        coef_df = pd.DataFrame({
            "feature": ds.feature_names,
            "coef_ckdu_ovr": ckdu_coef,
            "abs_coef": np.abs(ckdu_coef),
        }).sort_values("abs_coef", ascending=False)
        coef_df.to_csv(os.path.join(outdir, "fisher_lda_ckdu_coefficients.csv"), index=False)
        print("\n[Fisher/LDA] Top features by |LDA coef| for CKDu (one-vs-rest):")
        print(coef_df.head(12).to_string(index=False))

    # 2D LDA projection
    try:
        scaler: StandardScaler = pipe.named_steps["scaler"]
        Xs = scaler.transform(X)
        X_lda = lda.transform(Xs)
        if X_lda.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(7.0, 6.2))
            for lab in labels_sorted:
                idx = y == lab
                ax.scatter(X_lda[idx, 0], X_lda[idx, 1], s=28, alpha=0.8,
                           label=LABEL_MAP.get(int(lab), str(lab)))
            ax.set_xlabel("LD1")
            ax.set_ylabel("LD2")
            ax.set_title("LDA projection: LD1 vs LD2")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", frameon=True)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fisher_lda_scatter_ld1_ld2.png"), dpi=180)
            plt.close(fig)
    except Exception:
        pass

    print(f"\n[Fisher/LDA] Saved plots/tables to: {outdir}")


# ===================================
# 4) Kurtosis vs skewness (per feature)
# ===================================
def analysis_skew_kurt(ds: Dataset, outdir: str):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    Xs = StandardScaler().fit_transform(X)

    def skew_kurt(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sk = stats.skew(mat, axis=0, bias=False, nan_policy="omit")
        ku = stats.kurtosis(mat, axis=0, fisher=True, bias=False, nan_policy="omit")
        return sk, ku

    sk_all, ku_all = skew_kurt(Xs)
    df_all = pd.DataFrame({"feature": ds.feature_names, "skew": sk_all, "excess_kurtosis": ku_all})
    df_all.to_csv(os.path.join(outdir, "skew_kurtosis_all.csv"), index=False)

    if 1 in np.unique(y):
        sk_ck, ku_ck = skew_kurt(Xs[y == 1])
        sk_rest, ku_rest = skew_kurt(Xs[y != 1])

        delta = pd.DataFrame({
            "feature": ds.feature_names,
            "delta_skew_ckdu_minus_rest": sk_ck - sk_rest,
            "delta_kurtosis_ckdu_minus_rest": ku_ck - ku_rest,
        })
        delta["delta_norm"] = np.sqrt(delta["delta_skew_ckdu_minus_rest"] ** 2 +
                                      delta["delta_kurtosis_ckdu_minus_rest"] ** 2)
        delta = delta.sort_values("delta_norm", ascending=False)
        delta.to_csv(os.path.join(outdir, "skew_kurtosis_delta_ckdu_vs_rest.csv"), index=False)

        print("\n[Skew/Kurt] Top features with biggest distribution-shape shift (CKDu vs rest):")
        print(delta.head(12).to_string(index=False))

    def scatter(sk: np.ndarray, ku: np.ndarray, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        ax.scatter(sk, ku, s=38, alpha=0.85)
        ax.axvline(0, linewidth=1)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Skewness")
        ax.set_ylabel("Excess kurtosis")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        idx = np.argsort(np.abs(sk) + np.abs(ku))[-8:]
        for i in idx:
            ax.annotate(ds.feature_names[i], (sk[i], ku[i]), fontsize=8,
                        xytext=(4, 2), textcoords="offset points")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, filename), dpi=180)
        plt.close(fig)

    scatter(sk_all, ku_all, "Skewness vs kurtosis (all samples)", "skew_kurtosis_scatter_all.png")
    print(f"\n[Skew/Kurt] Saved plots/tables to: {outdir}")


# ==================================================
# 5) Feature-wise statistical significance
# ==================================================
def analysis_significance(ds: Dataset, outdir: str):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    feats = ds.feature_names

    labels_sorted = sorted(np.unique(y))
    groups = [X[y == lab] for lab in labels_sorted]

    anova_p, kruskal_p = [], []
    for j in range(X.shape[1]):
        cols = [g[:, j] for g in groups]
        anova_p.append(stats.f_oneway(*cols).pvalue)
        kruskal_p.append(stats.kruskal(*cols).pvalue)

    df_mc = pd.DataFrame({
        "feature": feats,
        "anova_p": anova_p,
        "anova_neglog10p": -np.log10(np.maximum(anova_p, 1e-300)),
        "kruskal_p": kruskal_p,
        "kruskal_neglog10p": -np.log10(np.maximum(kruskal_p, 1e-300)),
    }).sort_values("anova_p")
    df_mc.to_csv(os.path.join(outdir, "significance_multiclass_anova_kruskal.csv"), index=False)

    print("\n[Significance] Top features by ANOVA across 5 classes (smallest p):")
    print(df_mc.head(12).to_string(index=False))

    if 1 in np.unique(y):
        x1, x0 = X[y == 1], X[y != 1]
        t_p, mw_p = [], []
        for j in range(X.shape[1]):
            t_p.append(stats.ttest_ind(x1[:, j], x0[:, j], equal_var=False).pvalue)
            mw_p.append(stats.mannwhitneyu(x1[:, j], x0[:, j], alternative="two-sided").pvalue)

        df_bin = pd.DataFrame({
            "feature": feats,
            "t_p": t_p,
            "t_neglog10p": -np.log10(np.maximum(t_p, 1e-300)),
            "mw_p": mw_p,
            "mw_neglog10p": -np.log10(np.maximum(mw_p, 1e-300)),
        }).sort_values("t_p")
        df_bin.to_csv(os.path.join(outdir, "significance_ckdu_vs_rest_ttest_mwu.csv"), index=False)

        print("\n[Significance] Top features by Welch t-test (CKDu vs rest):")
        print(df_bin.head(12).to_string(index=False))

    # Plot -log10(p) (ANOVA)
    plot_df = df_mc.sort_values("anova_neglog10p", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(np.arange(len(plot_df)), plot_df["anova_neglog10p"].to_numpy())
    ax.set_xlabel("Features (sorted)")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title("Feature significance across 5 classes (ANOVA)")
    ax.grid(True, axis="y", alpha=0.25)
    for i in range(min(12, len(plot_df))):
        ax.text(i, plot_df.loc[i, "anova_neglog10p"] + 0.05, plot_df.loc[i, "feature"],
                rotation=90, fontsize=8, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "significance_anova_neglog10p.png"), dpi=180)
    plt.close(fig)

    print(f"\n[Significance] Saved plots/tables to: {outdir}")


# ======================================
# 6) Linear + Logistic regression analysis
# ======================================
def analysis_regression(ds: Dataset, outdir: str, test_size: float = 0.25):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            C=1.0,
            random_state=RANDOM_STATE,
        )),
    ])
    logreg.fit(Xtr, ytr)
    yhat = logreg.predict(Xte)
    proba = logreg.predict_proba(Xte)

    metrics = _basic_metrics(yte, yhat)
    try:
        metrics["auc_macro_ovr"] = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc_macro_ovr"] = float("nan")

    with open(os.path.join(outdir, "logreg_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "logreg_confusion_matrix.png"),
        title="Multinomial Logistic Regression (test set)",
    )

    print("\n[Regression] Logistic regression metrics (test split):")
    print(json.dumps(metrics, indent=2))

    clf: LogisticRegression = logreg.named_steps["clf"]
    coefs = clf.coef_
    classes = clf.classes_

    coef_df = pd.DataFrame(coefs, index=[LABEL_MAP[int(c)] for c in classes], columns=ds.feature_names)
    coef_df.to_csv(os.path.join(outdir, "logreg_coefficients_all_classes.csv"))

    if 1 in classes:
        ckdu_idx = np.where(classes == 1)[0][0]
        ck = pd.DataFrame({
            "feature": ds.feature_names,
            "coef_ckdu": coefs[ckdu_idx],
            "abs_coef": np.abs(coefs[ckdu_idx]),
        }).sort_values("abs_coef", ascending=False)
        ck.to_csv(os.path.join(outdir, "logreg_coefficients_ckdu.csv"), index=False)
        print("\n[Regression] Top features by |logistic coef| for CKDu class:")
        print(ck.head(12).to_string(index=False))

    # Linear regression baseline (Ridge) for CKDu vs rest
    if 1 in np.unique(y):
        y_bin = (y == 1).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_bin, test_size=test_size, random_state=RANDOM_STATE, stratify=y_bin
        )
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ])
        ridge.fit(Xtr, ytr)
        yscore = ridge.predict(Xte)
        yhat = (yscore >= 0.5).astype(int)
        metrics_r = {
            "accuracy": float(accuracy_score(yte, yhat)),
            "f1": float(f1_score(yte, yhat)),
            "auc": float(roc_auc_score(yte, yscore)),
        }
        with open(os.path.join(outdir, "ridge_ckdu_vs_rest_metrics.json"), "w") as f:
            json.dump(metrics_r, f, indent=2)

        w = ridge.named_steps["reg"].coef_
        w_df = pd.DataFrame({
            "feature": ds.feature_names,
            "ridge_w": w,
            "abs_w": np.abs(w),
        }).sort_values("abs_w", ascending=False)
        w_df.to_csv(os.path.join(outdir, "ridge_ckdu_vs_rest_coefficients.csv"), index=False)

        print("\n[Regression] Ridge (CKDu vs rest) metrics (test split):")
        print(json.dumps(metrics_r, indent=2))
        print("\n[Regression] Top features by |ridge coefficient| (CKDu vs rest):")
        print(w_df.head(12).to_string(index=False))

    print(f"\n[Regression] Saved plots/tables to: {outdir}")


# ======================
# 7) Gaussian Mixture Models
# ======================
def analysis_gmm(ds: Dataset, outdir: str, n_components_max: int = 3, test_size: float = 0.25):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # One GMM per class, BIC choose k in [1..n_components_max]
    # Diagonal covariance + single init keeps it fast and stable in 32D with limited samples.
    gmm_by_class: Dict[int, GaussianMixture] = {}
    bic_rows = []
    for lab in labels_sorted:
        Xc = Xtr_s[ytr == lab]
        best_gmm = None
        best_bic = np.inf
        best_k = 1
        for k in range(1, n_components_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=RANDOM_STATE,
                n_init=1,
            )
            gmm.fit(Xc)
            bic = gmm.bic(Xc)
            bic_rows.append((lab, k, bic))
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_k = k
        gmm_by_class[int(lab)] = best_gmm
        print(f"[GMM] Class {LABEL_MAP[int(lab)]}: best #components by BIC = {best_k}")

    bic_df = pd.DataFrame(bic_rows, columns=["class_label", "n_components", "bic"])
    bic_df["class_name"] = bic_df["class_label"].map(LABEL_MAP)
    bic_df.to_csv(os.path.join(outdir, "gmm_bic_grid.csv"), index=False)

    # Predict by max log posterior: log p(x|k) + log prior(k)
    priors = {int(l): float(np.mean(ytr == l)) for l in labels_sorted}
    log_prior = np.array([np.log(priors[int(l)]) for l in labels_sorted])

    loglik = np.zeros((len(Xte_s), len(labels_sorted)), dtype=float)
    for j, lab in enumerate(labels_sorted):
        loglik[:, j] = gmm_by_class[int(lab)].score_samples(Xte_s)
    logpost = loglik + log_prior
    yhat = np.array([labels_sorted[i] for i in np.argmax(logpost, axis=1)], dtype=int)

    metrics = _basic_metrics(yte, yhat)
    with open(os.path.join(outdir, "gmm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "gmm_confusion_matrix.png"),
        title="GMM classifier (one diag-GMM per class, test set)",
    )
    print("\n[GMM] Metrics (test split):")
    print(json.dumps(metrics, indent=2))

    # Feature importance for CKDu: effect size using GMM moments (fast + interpretable)
    if 1 in np.unique(y):
        def gmm_moments_diag(g: GaussianMixture) -> Tuple[np.ndarray, np.ndarray]:
            w = g.weights_
            mu = g.means_
            var = g.covariances_
            mean = (w[:, None] * mu).sum(axis=0)
            second = (w[:, None] * (var + mu ** 2)).sum(axis=0)
            variance = np.maximum(second - mean ** 2, 1e-12)
            return mean, variance

        mu_ck, var_ck = gmm_moments_diag(gmm_by_class[1])
        other_labels = [int(l) for l in labels_sorted if int(l) != 1]
        w_other = np.array([priors[l] for l in other_labels], dtype=float)
        w_other /= w_other.sum()

        mu_rest = np.zeros_like(mu_ck)
        sec_rest = np.zeros_like(mu_ck)
        for w, lab in zip(w_other, other_labels):
            mu_c, var_c = gmm_moments_diag(gmm_by_class[lab])
            mu_rest += w * mu_c
            sec_rest += w * (var_c + mu_c ** 2)
        var_rest = np.maximum(sec_rest - mu_rest ** 2, 1e-12)

        d = (mu_ck - mu_rest) / np.sqrt(0.5 * (var_ck + var_rest))
        eff_df = pd.DataFrame({
            "feature": ds.feature_names,
            "gmm_effect_size_ckdu_vs_rest": d,
            "abs_effect_size": np.abs(d),
        }).sort_values("abs_effect_size", ascending=False)
        eff_df.to_csv(os.path.join(outdir, "gmm_feature_effect_size_ckdu_vs_rest.csv"), index=False)

        print("\n[GMM] Top features by |effect size| (CKDu vs rest), derived from GMM moments:")
        print(eff_df.head(12).to_string(index=False))

    print(f"\n[GMM] Saved plots/tables to: {outdir}")


# ======================
# 8) Neural Networks (MLP)
# ======================
def analysis_nn(ds: Dataset, outdir: str, test_size: float = 0.25):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )),
    ])
    mlp.fit(Xtr, ytr)
    yhat = mlp.predict(Xte)
    proba = mlp.predict_proba(Xte)

    metrics = _basic_metrics(yte, yhat)
    try:
        metrics["auc_macro_ovr"] = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc_macro_ovr"] = float("nan")

    with open(os.path.join(outdir, "mlp_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "mlp_confusion_matrix.png"),
        title="MLP neural network (test set)",
    )

    print("\n[NN/MLP] Metrics (test split):")
    print(json.dumps(metrics, indent=2))
    print("\n[NN/MLP] Classification report:")
    print(classification_report(yte, yhat, target_names=label_names, digits=3))

    # Permutation importance (model-agnostic)
    scorer = make_scorer(f1_score, average="macro")
    imp = permutation_importance(
        mlp,
        Xte,
        yte,
        scoring=scorer,
        n_repeats=8,
        random_state=RANDOM_STATE,
        # Avoid multiprocessing in restrictive environments (can cause EOF/worker issues).
        n_jobs=1,
    )
    imp_df = pd.DataFrame({
        "feature": ds.feature_names,
        "perm_importance_mean": imp.importances_mean,
        "perm_importance_std": imp.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    imp_df.to_csv(os.path.join(outdir, "mlp_permutation_importance.csv"), index=False)

    print("\n[NN/MLP] Top features by permutation importance (macro-F1 drop):")
    print(imp_df.head(12).to_string(index=False))

    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.barh(top["feature"], top["perm_importance_mean"], xerr=top["perm_importance_std"])
    ax.set_xlabel("Permutation importance (mean ± std)")
    ax.set_title("MLP permutation feature importance (top 15)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mlp_permutation_importance_top15.png"), dpi=180)
    plt.close(fig)

    print(f"\n[NN/MLP] Saved plots/tables to: {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/data/CKDu_processed.csv", help="Path to CKDu_processed.csv")
    ap.add_argument(
        "--analysis",
        required=True,
        choices=["pca", "bayes", "fisher", "skew_kurt", "significance", "regression", "gmm", "nn"],
        help="Which analysis to run",
    )
    ap.add_argument("--outdir", default="/mnt/data/ckdu_results", help="Output directory")
    ap.add_argument("--no_age_scr", action="store_true", help="Exclude Age and SCR (use only F1..F30)")
    args = ap.parse_args()

    ds = load_ckdu_processed(args.csv, use_age_scr=(not args.no_age_scr))
    outdir = os.path.join(args.outdir, args.analysis)

    if args.analysis == "pca":
        analysis_pca(ds, outdir=outdir)
    elif args.analysis == "bayes":
        analysis_bayes(ds, outdir=outdir)
    elif args.analysis == "fisher":
        analysis_fisher(ds, outdir=outdir)
    elif args.analysis == "skew_kurt":
        analysis_skew_kurt(ds, outdir=outdir)
    elif args.analysis == "significance":
        analysis_significance(ds, outdir=outdir)
    elif args.analysis == "regression":
        analysis_regression(ds, outdir=outdir)
    elif args.analysis == "gmm":
        analysis_gmm(ds, outdir=outdir)
    elif args.analysis == "nn":
        analysis_nn(ds, outdir=outdir)
    else:
        raise ValueError("Unknown analysis")


if __name__ == "__main__":
    main()
