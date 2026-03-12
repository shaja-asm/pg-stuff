#!/usr/bin/env python3
"""
CKDu multi-class analysis suite with metal-name mapping, PCA-guided fusion,
and explainable AI support (SHAP + LIME).

Dataset assumptions
-------------------
- CSV has no header row
- Column 0: class label in {1..5}
- Remaining columns are numeric features. This script uses the 30 metal
  features only, mapped directly (in order) to:
    Na, Mg, K, Ca, Li, Be, Al, V, Cr, Mn, Fe, Co, Ni, Cu, Zn,
    Ga, As, Se, Rb, Sr, Ag, Cd, In, Cs, Ba, Hg, Tl, Pb, Bi, U

Key additions in this version
-----------------------------
1) Replaces F1..F30 with the actual metal names.
2) Adds a PCA-guided fusion model based on the strongest methods from the presentation:
      - LDA
      - Logistic Regression
      - MLP
3) Handles class imbalance in the fusion model.
4) Adds SHAP and LIME explanations focused on CKDu.
5) Adds an `all` option to run the full pipeline in one command.
6) Adds GenAI discussion notes as a markdown output for report writing.

Recommended usage
-----------------
Run the full workflow:
  python ckdu_analysis_suite_v4_fusion.py --csv CKDu_processed.csv --analysis all

Run only the PCA-guided fusion + XAI:
  python ckdu_analysis_suite_v4_fusion.py --csv CKDu_processed.csv --analysis fusion

Optional packages for full XAI support:
  pip install shap lime
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import textwrap
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
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

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

LABEL_MAP = {
    1: "CKDu",
    2: "EC",
    3: "NEC",
    4: "ECKD",
    5: "NECKD",
}

METAL_FEATURE_NAMES = [
    "Na", "Mg", "K", "Ca", "Li", "Be", "Al", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Se", "Rb", "Sr",
    "Ag", "Cd", "In", "Cs", "Ba", "Hg", "Tl", "Pb", "Bi", "U",
]

F_ALIAS_MAP = {f"F{i + 1}": name for i, name in enumerate(METAL_FEATURE_NAMES)}
REVERSE_F_ALIAS_MAP = {v: k for k, v in F_ALIAS_MAP.items()}


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    class_names: List[str]


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _make_feature_alias_lookup(feature_names: Iterable[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for feat in feature_names:
        lookup[_norm_name(feat)] = feat
        if feat in REVERSE_F_ALIAS_MAP:
            lookup[_norm_name(REVERSE_F_ALIAS_MAP[feat])] = feat
    return lookup


def resolve_feature_names(requested_features: List[str], feature_names: List[str]) -> List[str]:
    lookup = _make_feature_alias_lookup(feature_names)
    resolved: List[str] = []
    missing: List[str] = []
    for item in requested_features:
        key = _norm_name(item)
        if key in lookup:
            resolved.append(lookup[key])
        else:
            missing.append(item)
    if missing:
        raise ValueError(
            f"Could not resolve these feature names: {missing}. "
            f"Use actual names such as {feature_names[:8]} or aliases like F1..F30."
        )
    return resolved


def load_ckdu_processed(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 1 + len(METAL_FEATURE_NAMES):
        raise ValueError(
            f"Expected at least {1 + len(METAL_FEATURE_NAMES)} columns "
            f"(label + {len(METAL_FEATURE_NAMES)} metals), got {df.shape[1]}"
        )

    y = df.iloc[:, 0].astype(int).to_numpy()
    mask = np.isin(y, list(LABEL_MAP.keys()))
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]

    feature_df = df.iloc[:, 1:].copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    # Strict metals-only policy:
    # - If layout is [Age, S.cr, 30 metals, ...], always drop Age/S.cr.
    # - If layout is already 30 metals, use those directly.
    # - Any ambiguous layout raises an error.
    n_features = feature_df.shape[1]
    if n_features == len(METAL_FEATURE_NAMES):
        metal_df = feature_df.iloc[:, :len(METAL_FEATURE_NAMES)].copy()
    elif n_features >= len(METAL_FEATURE_NAMES) + 2:
        metal_df = feature_df.iloc[:, 2:2 + len(METAL_FEATURE_NAMES)].copy()
    else:
        raise ValueError(
            "Unsupported feature-column count after label: "
            f"{n_features}. Expected either 30 metals-only columns, "
            "or at least 32 columns including Age/S.cr + 30 metals."
        )

    if metal_df.shape[1] != len(METAL_FEATURE_NAMES):
        raise ValueError(
            f"Expected exactly {len(METAL_FEATURE_NAMES)} metal features after filtering, "
            f"got {metal_df.shape[1]}."
        )

    metal_df.columns = METAL_FEATURE_NAMES

    X = metal_df.to_numpy(dtype=float)
    final_feature_names = list(metal_df.columns)
    class_names = [LABEL_MAP[i] for i in sorted(np.unique(y))]
    return Dataset(X=X, y=y, feature_names=final_feature_names, class_names=class_names)


# =========================
# Shared helpers / utilities
# =========================

def _make_multinomial_logreg(class_weight=None) -> LogisticRegression:
    kwargs = dict(
        solver="lbfgs",
        max_iter=5000,
        C=0.8,
        tol=1e-4,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs)


def _make_regularized_lda(priors: Optional[np.ndarray] = None) -> LinearDiscriminantAnalysis:
    # Use shrinkage-enabled LDA to stabilize covariance estimates.
    return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", priors=priors)


def _make_regularized_mlp(
        hidden_layer_sizes: Tuple[int, int],
        random_state: int,
        max_iter: int = 1600,
        n_iter_no_change: int = 25,
) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=3e-4,
        learning_rate_init=1e-3,
        learning_rate="adaptive",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, low_q: float = 0.01, high_q: float = 0.99):
        self.low_q = low_q
        self.high_q = high_q

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_ = np.nanquantile(X, self.low_q, axis=0)
        self.upper_ = np.nanquantile(X, self.high_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)


class PowerTransformerSafe(BaseEstimator, TransformerMixin):
    """Yeo-Johnson that gracefully handles near-constant columns."""

    def __init__(self):
        self.transformers_: List[Optional[PowerTransformer]] = []
        self.constant_values_: List[float] = []

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.transformers_ = []
        self.constant_values_ = []
        for j in range(X.shape[1]):
            col = X[:, [j]]
            finite_col = col[np.isfinite(col)]
            if finite_col.size == 0:
                self.transformers_.append(None)
                self.constant_values_.append(0.0)
                continue
            if np.nanstd(finite_col) < 1e-12:
                self.transformers_.append(None)
                self.constant_values_.append(float(np.nanmedian(finite_col)))
                continue
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            pt.fit(col)
            self.transformers_.append(pt)
            self.constant_values_.append(float(np.nanmedian(finite_col)))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xt = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            col = X[:, [j]]
            pt = self.transformers_[j]
            if pt is None:
                fill = self.constant_values_[j]
                col = np.where(np.isfinite(col), col, fill)
                Xt[:, j] = col.ravel()
            else:
                Xt[:, j] = pt.transform(col).ravel()
        return Xt


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)


def make_preprocessor(kind: str = "robust") -> Pipeline:
    kind = kind.lower().strip()
    if kind == "robust":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", QuantileClipper(low_q=0.01, high_q=0.99)),
            ("power", PowerTransformerSafe()),
            ("scaler", RobustScaler(quantile_range=(25.0, 75.0))),
        ])
    if kind == "standard":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    raise ValueError(f"Unknown preprocessor kind: {kind}")


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _safe_multiclass_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

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

def _save_json(obj: dict, outpath: str):
    with open(outpath, "w") as f:
        json.dump(obj, f, indent=2)

def _neglog10p(pvals: np.ndarray) -> np.ndarray:
    return -np.log10(np.maximum(np.asarray(pvals, dtype=float), 1e-300))

def random_oversample(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE) -> Tuple[
    np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))
    indices: List[np.ndarray] = []
    for c, cnt in zip(classes, counts):
        idx = np.flatnonzero(y == c)
        if cnt < max_count:
            extra = rng.choice(idx, size=max_count - cnt, replace=True)
            idx = np.concatenate([idx, extra])
        indices.append(idx)
    all_idx = np.concatenate(indices)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]

def _uniform_priors(classes: np.ndarray) -> np.ndarray:
    return np.ones(len(classes), dtype=float) / float(len(classes))

def _align_proba(proba: np.ndarray, model_classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    for j, c in enumerate(target_classes):
        k = np.where(model_classes == c)[0]
        if len(k):
            aligned[:, j] = proba[:, k[0]]
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return aligned / row_sum

def _top_bar_plot(df: pd.DataFrame, value_col: str, title: str, outpath: str, top_n: int = 15):
    plot_df = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.barh(plot_df["feature"], plot_df[value_col])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _derive_frequency_consensus_features_from_outputs(
        root_outdir: str,
        feature_names: List[str],
        top_k: int = 8,
        source_top_n: int = 8,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    # Each source contributes its top-N features; consensus is by appearance frequency.
    source_specs = [
        ("pca_sep", "pca/pca_ckdu_separation_feature_scores.csv", "abs_score", False),
        ("bayes_single", "bayes/bayes_single_feature_ranking.csv", "cv_f1_macro", False),
        ("bayes_loo", "bayes/bayes_leave_one_out_importance.csv", "f1_drop_when_removed", False),
        ("bayes_llr", "bayes/bayes_llr_feature_contributions.csv", "avg_abs_llr_vs_each_class", False),
        ("fisher_coef", "fisher/fisher_lda_ckdu_coefficients.csv", "abs_coef", False),
        ("skew_kurt_delta", "skew_kurt/skew_kurtosis_delta_ckdu_vs_rest.csv", "delta_norm", False),
        ("signif_anova", "significance/significance_multiclass_anova_kruskal.csv", "anova_neglog10p", False),
        ("signif_ttest", "significance/significance_ckdu_vs_rest_ttest_mwu.csv", "t_neglog10p", False),
        ("logreg_ckdu", "regression/logreg_coefficients_ckdu.csv", "abs_coef", False),
        ("ridge_ckdu", "regression/ridge_ckdu_vs_rest_coefficients.csv", "abs_w", False),
        ("gmm_effect", "gmm/gmm_feature_effect_size_ckdu_vs_rest.csv", "abs_effect_size", False),
        ("nn_perm", "nn/mlp_permutation_importance.csv", "perm_importance_mean", False),
    ]

    name_set = set(feature_names)
    counts = {f: 0 for f in feature_names}
    rank_sums = {f: 0.0 for f in feature_names}
    hit_rows: List[Dict[str, object]] = []

    for source_name, rel_path, score_col, ascending in source_specs:
        path = os.path.join(root_outdir, rel_path)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "feature" not in df.columns or score_col not in df.columns:
            continue

        source_df = df[["feature", score_col]].copy()
        source_df["feature"] = source_df["feature"].astype(str)
        source_df = source_df[source_df["feature"].isin(name_set)]
        source_df = source_df.dropna(subset=[score_col])
        if source_df.empty:
            continue

        source_df = source_df.sort_values(score_col, ascending=ascending).head(max(1, int(source_top_n)))
        for rank, row in enumerate(source_df.itertuples(index=False), start=1):
            feat = str(getattr(row, "feature"))
            score = float(getattr(row, score_col))
            counts[feat] += 1
            rank_sums[feat] += float(rank)
            hit_rows.append({
                "source": source_name,
                "feature": feat,
                "rank_in_source": rank,
                "source_score": score,
            })

    summary_df = pd.DataFrame({
        "feature": feature_names,
        "appearance_count": [counts[f] for f in feature_names],
        "avg_rank_if_hit": [
            (rank_sums[f] / counts[f]) if counts[f] > 0 else np.nan
            for f in feature_names
        ],
        "legacy_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in feature_names],
    }).sort_values(
        ["appearance_count", "avg_rank_if_hit", "feature"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    selected = summary_df.loc[summary_df["appearance_count"] > 0, "feature"].head(max(1, int(top_k))).tolist()
    hits_df = pd.DataFrame(hit_rows)
    return selected, summary_df, hits_df

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

    loadings = pca.components_.T
    load_df = pd.DataFrame(loadings, index=ds.feature_names,
                           columns=[f"PC{i}" for i in range(1, loadings.shape[1] + 1)])
    load_df.to_csv(os.path.join(outdir, "pca_loadings.csv"))

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
            "legacy_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in ds.feature_names],
        }).sort_values("abs_score", ascending=False)
        sep_df.to_csv(os.path.join(outdir, "pca_ckdu_separation_feature_scores.csv"), index=False)
        print("\n[PCA] Top features driving CKDu centroid separation (|score|):")
        print(sep_df.head(12).to_string(index=False))

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
    metrics["auc_macro_ovr"] = _safe_multiclass_auc(y, y_proba_oof)
    _save_json(metrics, os.path.join(outdir, "bayes_cv_metrics.json"))

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
        "anova_neglog10p": _neglog10p(anova_p),
        "kruskal_p": kruskal_p,
        "kruskal_neglog10p": _neglog10p(kruskal_p),
    }).sort_values("anova_p")
    df_mc.to_csv(os.path.join(outdir, "significance_multiclass_anova_kruskal.csv"), index=False)

    if 1 in np.unique(y):
        x1, x0 = X[y == 1], X[y != 1]
        t_p, mw_p = [], []
        for j in range(X.shape[1]):
            t_p.append(stats.ttest_ind(x1[:, j], x0[:, j], equal_var=False).pvalue)
            mw_p.append(stats.mannwhitneyu(x1[:, j], x0[:, j], alternative="two-sided").pvalue)

        df_bin = pd.DataFrame({
            "feature": feats,
            "t_p": t_p,
            "t_neglog10p": _neglog10p(t_p),
            "mw_p": mw_p,
            "mw_neglog10p": _neglog10p(mw_p),
        }).sort_values("t_p")
        df_bin.to_csv(os.path.join(outdir, "significance_ckdu_vs_rest_ttest_mwu.csv"), index=False)

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
        ("clf", _make_multinomial_logreg(class_weight="balanced")),
    ])
    logreg.fit(Xtr, ytr)
    yhat = logreg.predict(Xte)
    proba = logreg.predict_proba(Xte)

    metrics = _basic_metrics(yte, yhat)
    metrics["auc_macro_ovr"] = _safe_multiclass_auc(yte, proba)
    _save_json(metrics, os.path.join(outdir, "logreg_metrics.json"))

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "logreg_confusion_matrix.png"),
        title="Multinomial Logistic Regression (test set)",
    )

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

    if 1 in np.unique(y):
        y_bin = (y == 1).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_bin, test_size=test_size, random_state=RANDOM_STATE, stratify=y_bin
        )
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.2, random_state=RANDOM_STATE)),
        ])
        ridge.fit(Xtr, ytr)
        yscore = ridge.predict(Xte)
        yhat = (yscore >= 0.5).astype(int)
        metrics_r = {
            "accuracy": float(accuracy_score(yte, yhat)),
            "f1": float(f1_score(yte, yhat)),
            "auc": float(roc_auc_score(yte, yscore)),
        }
        _save_json(metrics_r, os.path.join(outdir, "ridge_ckdu_vs_rest_metrics.json"))

        w = ridge.named_steps["reg"].coef_
        w_df = pd.DataFrame({
            "feature": ds.feature_names,
            "ridge_w": w,
            "abs_w": np.abs(w),
        }).sort_values("abs_w", ascending=False)
        w_df.to_csv(os.path.join(outdir, "ridge_ckdu_vs_rest_coefficients.csv"), index=False)

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
                n_init=3,
                reg_covar=3e-5,
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

    priors = {int(l): float(np.mean(ytr == l)) for l in labels_sorted}
    log_prior = np.array([np.log(priors[int(l)]) for l in labels_sorted])

    loglik = np.zeros((len(Xte_s), len(labels_sorted)), dtype=float)
    for j, lab in enumerate(labels_sorted):
        loglik[:, j] = gmm_by_class[int(lab)].score_samples(Xte_s)
    logpost = loglik + log_prior
    yhat = np.array([labels_sorted[i] for i in np.argmax(logpost, axis=1)], dtype=int)

    metrics = _basic_metrics(yte, yhat)
    _save_json(metrics, os.path.join(outdir, "gmm_metrics.json"))

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "gmm_confusion_matrix.png"),
        title="GMM classifier (one diag-GMM per class, test set)",
    )

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
        ("clf", _make_regularized_mlp(
            hidden_layer_sizes=(32, 16),
            random_state=RANDOM_STATE,
            max_iter=1500,
            n_iter_no_change=20,
        )),
    ])
    mlp.fit(Xtr, ytr)
    yhat = mlp.predict(Xte)
    proba = mlp.predict_proba(Xte)

    metrics = _basic_metrics(yte, yhat)
    metrics["auc_macro_ovr"] = _safe_multiclass_auc(yte, proba)
    _save_json(metrics, os.path.join(outdir, "mlp_metrics.json"))

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "mlp_confusion_matrix.png"),
        title="MLP neural network (test set)",
    )

    scorer = make_scorer(f1_score, average="macro")
    imp = permutation_importance(
        mlp,
        Xte,
        yte,
        scoring=scorer,
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    imp_df = pd.DataFrame({
        "feature": ds.feature_names,
        "perm_importance_mean": imp.importances_mean,
        "perm_importance_std": imp.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    imp_df.to_csv(os.path.join(outdir, "mlp_permutation_importance.csv"), index=False)
    _top_bar_plot(
        imp_df,
        value_col="perm_importance_mean",
        title="MLP permutation feature importance (top 15)",
        outpath=os.path.join(outdir, "mlp_permutation_importance_top15.png"),
        top_n=15,
    )

    print(f"\n[NN/MLP] Saved plots/tables to: {outdir}")

# ==============================================================
# 9) Consensus validation: top-k features + robust preprocessing
# ==============================================================
def analysis_validate_top_features(
        ds: Dataset,
        outdir: str,
        top_features: List[str],
        preproc_kind: str = "robust",
        n_splits: int = 5,
        perm_repeats: int = 10,
):
    outdir = _ensure_outdir(outdir)

    top_features = resolve_feature_names(top_features, ds.feature_names)
    name_to_idx = {n: i for i, n in enumerate(ds.feature_names)}
    col_idx = [name_to_idx[f] for f in top_features]
    X = ds.X[:, col_idx]
    y = ds.y

    _save_json(
        {
            "top_features": top_features,
            "top_features_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in top_features],
            "preproc_kind": preproc_kind,
            "n_splits": n_splits,
            "perm_repeats": perm_repeats,
        },
        os.path.join(outdir, "consensus_features_used.json"),
    )

    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    preproc = make_preprocessor(preproc_kind)

    logreg_pipe = Pipeline([
        ("pre", preproc),
        ("clf", _make_multinomial_logreg(class_weight="balanced")),
    ])

    oof_pred_lr = np.zeros_like(y)
    oof_proba_lr = np.zeros((len(y), len(labels_sorted)), dtype=float)
    fold_metrics_lr = []
    coef_by_fold = []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = clone(logreg_pipe)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])

        classes = model.named_steps["clf"].classes_
        proba_aligned = _align_proba(proba, classes, np.asarray(labels_sorted))

        oof_pred_lr[te] = pred
        oof_proba_lr[te, :] = proba_aligned

        m = _basic_metrics(y[te], pred)
        m["auc_macro_ovr"] = _safe_multiclass_auc(y[te], proba_aligned)
        m["fold"] = fold
        fold_metrics_lr.append(m)

        clf: LogisticRegression = model.named_steps["clf"]
        if 1 in clf.classes_:
            ckdu_idx = int(np.where(clf.classes_ == 1)[0][0])
            coef_by_fold.append(clf.coef_[ckdu_idx].copy())

    lr_metrics_df = pd.DataFrame(fold_metrics_lr)
    lr_metrics_df.to_csv(os.path.join(outdir, "logreg_cv_fold_metrics.csv"), index=False)

    lr_summary = {
        k: {
            "mean": float(np.nanmean(lr_metrics_df[k].to_numpy())),
            "std": float(np.nanstd(lr_metrics_df[k].to_numpy(), ddof=1)),
        }
        for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc_macro_ovr"]
        if k in lr_metrics_df.columns
    }
    _save_json(lr_summary, os.path.join(outdir, "logreg_cv_summary.json"))

    _save_confusion(
        y_true=y,
        y_pred=oof_pred_lr,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "logreg_oof_confusion_matrix.png"),
        title=f"LogReg + {preproc_kind} preproc (OOF, {n_splits}-fold CV)",
    )

    with open(os.path.join(outdir, "logreg_oof_classification_report.txt"), "w") as f:
        f.write(classification_report(y, oof_pred_lr, target_names=label_names, digits=3))

    if coef_by_fold:
        coef_mat = np.vstack(coef_by_fold)
        coef_summary_df = pd.DataFrame({
            "feature": top_features,
            "coef_mean": coef_mat.mean(axis=0),
            "coef_std": coef_mat.std(axis=0, ddof=1),
            "abs_coef_mean": np.abs(coef_mat).mean(axis=0),
            "abs_coef_std": np.abs(coef_mat).std(axis=0, ddof=1),
        }).sort_values("abs_coef_mean", ascending=False)
        coef_summary_df.to_csv(os.path.join(outdir, "logreg_ckdu_coef_cv_summary.csv"), index=False)

    mlp_pipe = Pipeline([
        ("pre", preproc),
        ("clf", _make_regularized_mlp(
            hidden_layer_sizes=(16, 8),
            random_state=RANDOM_STATE,
            max_iter=2000,
            n_iter_no_change=25,
        )),
    ])

    oof_pred_mlp = np.zeros_like(y)
    oof_proba_mlp = np.zeros((len(y), len(labels_sorted)), dtype=float)
    fold_metrics_mlp = []
    imp_by_fold = []

    scorer = make_scorer(f1_score, average="macro")

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = clone(mlp_pipe)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])

        classes = model.named_steps["clf"].classes_
        proba_aligned = _align_proba(proba, classes, np.asarray(labels_sorted))

        oof_pred_mlp[te] = pred
        oof_proba_mlp[te, :] = proba_aligned

        m = _basic_metrics(y[te], pred)
        m["auc_macro_ovr"] = _safe_multiclass_auc(y[te], proba_aligned)
        m["fold"] = fold
        fold_metrics_mlp.append(m)

        imp = permutation_importance(
            model,
            X[te],
            y[te],
            scoring=scorer,
            n_repeats=perm_repeats,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        imp_by_fold.append(imp.importances_mean.copy())

    mlp_metrics_df = pd.DataFrame(fold_metrics_mlp)
    mlp_metrics_df.to_csv(os.path.join(outdir, "mlp_cv_fold_metrics.csv"), index=False)

    mlp_summary = {
        k: {
            "mean": float(np.nanmean(mlp_metrics_df[k].to_numpy())),
            "std": float(np.nanstd(mlp_metrics_df[k].to_numpy(), ddof=1)),
        }
        for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc_macro_ovr"]
        if k in mlp_metrics_df.columns
    }
    _save_json(mlp_summary, os.path.join(outdir, "mlp_cv_summary.json"))

    _save_confusion(
        y_true=y,
        y_pred=oof_pred_mlp,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "mlp_oof_confusion_matrix.png"),
        title=f"MLP + {preproc_kind} preproc (OOF, {n_splits}-fold CV)",
    )

    if imp_by_fold:
        imp_mat = np.vstack(imp_by_fold)
        imp_summary_df = pd.DataFrame({
            "feature": top_features,
            "perm_importance_mean": imp_mat.mean(axis=0),
            "perm_importance_std": imp_mat.std(axis=0, ddof=1),
        }).sort_values("perm_importance_mean", ascending=False)
        imp_summary_df.to_csv(os.path.join(outdir, "mlp_perm_importance_cv_summary.csv"), index=False)

    print("\n[Validate Top Features] Completed.")
    print(f"[Validate Top Features] Features used: {top_features}")
    print(f"[Validate Top Features] Saved outputs to: {outdir}")

# =============================================
# 10) PCA-guided fusion model + explainable AI
# =============================================
def _feature_selection_consensus(
        X_proc: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        raw_top_k: int,
) -> pd.DataFrame:
    """Select raw features using the strongest ideas from the presentation.

    This deliberately emphasizes the better-performing and more interpretable methods:
    - multi-class ANOVA
    - CKDu-vs-rest Welch t-test
    - LDA CKDu coefficients
    - Logistic Regression CKDu coefficients

    The final raw-feature shortlist is based on average rank across these views.
    """
    labels_sorted = sorted(np.unique(y))
    groups = [X_proc[y == lab] for lab in labels_sorted]

    anova_p = []
    for j in range(X_proc.shape[1]):
        cols = [g[:, j] for g in groups]
        anova_p.append(stats.f_oneway(*cols).pvalue)
    anova_score = _neglog10p(anova_p)

    if 1 in np.unique(y):
        x_ck = X_proc[y == 1]
        x_rest = X_proc[y != 1]
        t_p = [stats.ttest_ind(x_ck[:, j], x_rest[:, j], equal_var=False).pvalue for j in
               range(X_proc.shape[1])]
        ckdu_score = _neglog10p(t_p)
    else:
        ckdu_score = np.zeros(X_proc.shape[1])

    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(X_proc, y)
    if 1 in lda.classes_ and hasattr(lda, "coef_"):
        lda_coef = np.abs(lda.coef_[np.where(lda.classes_ == 1)[0][0]])
    else:
        lda_coef = np.zeros(X_proc.shape[1])

    logreg = _make_multinomial_logreg(class_weight="balanced")
    logreg.fit(X_proc, y)
    if 1 in logreg.classes_:
        log_coef = np.abs(logreg.coef_[np.where(logreg.classes_ == 1)[0][0]])
    else:
        log_coef = np.zeros(X_proc.shape[1])

    df = pd.DataFrame({
        "feature": feature_names,
        "anova_score": anova_score,
        "ckdu_t_score": ckdu_score,
        "lda_abs_coef": lda_coef,
        "logreg_abs_coef": log_coef,
    })
    for col in ["anova_score", "ckdu_t_score", "lda_abs_coef", "logreg_abs_coef"]:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="average")
    df["avg_rank"] = df[[
        "rank_anova_score", "rank_ckdu_t_score", "rank_lda_abs_coef", "rank_logreg_abs_coef"
    ]].mean(axis=1)
    df["selected_for_raw_branch"] = 0
    if raw_top_k > 0:
        selected_idx = df.nsmallest(raw_top_k, "avg_rank").index
        df.loc[selected_idx, "selected_for_raw_branch"] = 1
    return df.sort_values("avg_rank", ascending=True).reset_index(drop=True)

class PCAGuidedFusionClassifier(BaseEstimator, ClassifierMixin):
    """Fusion model using PCA as the main branch + a small raw-feature branch.

    Base models:
      - LDA on PCA features (strongest CV performer in the presentation)
      - class-balanced Logistic Regression on [PCA | selected raw features]
      - MLP on balanced resampled [PCA | selected raw features]

    Final probabilities are combined by weighted soft voting.
    """

    def __init__(
            self,
            preproc_kind: str = "robust",
            pca_variance_threshold: float = 0.95,
            max_pca_components: int = 12,
            raw_top_k: int = 8,
            internal_cv: int = 3,
            random_state: int = RANDOM_STATE,
            base_weights: Optional[List[float]] = None,
            mlp_hidden: Tuple[int, int] = (24, 12),
            feature_names: Optional[List[str]] = None,
            forced_raw_features: Optional[List[str]] = None,
    ):
        self.preproc_kind = preproc_kind
        self.pca_variance_threshold = pca_variance_threshold
        self.max_pca_components = max_pca_components
        self.raw_top_k = raw_top_k
        self.internal_cv = internal_cv
        self.random_state = random_state
        self.base_weights = base_weights
        self.mlp_hidden = mlp_hidden
        self.feature_names = feature_names
        self.forced_raw_features = forced_raw_features

    def _build_representation(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_proc = self.pre_.transform(X)
        X_pca = self.pca_.transform(X_proc)
        if len(self.selected_raw_idx_):
            X_raw = X_proc[:, self.selected_raw_idx_]
            X_fused = np.hstack([X_pca, X_raw])
        else:
            X_fused = X_pca.copy()
        return X_proc, X_pca, X_fused

    def _auto_weights(self, X_pca: np.ndarray, X_fused: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.base_weights is not None:
            arr = np.asarray(self.base_weights, dtype=float)
            arr = np.maximum(arr, 1e-9)
            return arr / arr.sum()

        skf = StratifiedKFold(n_splits=max(2, self.internal_cv), shuffle=True, random_state=self.random_state)
        scorer = make_scorer(f1_score, average="macro")
        score_map = {"lda": [], "logreg": [], "mlp": []}
        classes = np.sort(np.unique(y))

        for tr, te in skf.split(X_fused, y):
            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(X_pca[tr], y[tr])
            score_map["lda"].append(f1_score(y[te], lda.predict(X_pca[te]), average="macro"))

            logreg = _make_multinomial_logreg(class_weight="balanced")
            logreg.fit(X_fused[tr], y[tr])
            score_map["logreg"].append(f1_score(y[te], logreg.predict(X_fused[te]), average="macro"))

            Xb, yb = random_oversample(X_fused[tr], y[tr], random_state=self.random_state)
            mlp = _make_regularized_mlp(
                hidden_layer_sizes=self.mlp_hidden,
                random_state=self.random_state,
                max_iter=1600,
                n_iter_no_change=25,
            )
            mlp.fit(Xb, yb)
            score_map["mlp"].append(f1_score(y[te], mlp.predict(X_fused[te]), average="macro"))

        scores = np.array([
            np.mean(score_map["lda"]),
            np.mean(score_map["logreg"]),
            np.mean(score_map["mlp"]),
        ], dtype=float)
        scores = np.maximum(scores, 1e-9)
        return scores / scores.sum()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self.feature_names_in_ = list(self.feature_names) if self.feature_names is not None else [f"x{i}" for i
                                                                                                  in range(
                X.shape[1])]

        self.pre_ = make_preprocessor(self.preproc_kind)
        X_proc = self.pre_.fit_transform(X)

        self.selector_table_ = _feature_selection_consensus(
            X_proc=X_proc,
            y=y,
            feature_names=self.feature_names_in_,
            raw_top_k=self.raw_top_k,
        )
        name_to_idx = {n: i for i, n in enumerate(self.feature_names_in_)}
        if self.forced_raw_features:
            forced = resolve_feature_names(
                [s for s in self.forced_raw_features if str(s).strip()],
                self.feature_names_in_,
            )
            # Keep order, remove duplicates.
            selected = list(dict.fromkeys(forced))
            self.selector_table_["selected_for_raw_branch"] = 0
            self.selector_table_.loc[
                self.selector_table_["feature"].isin(selected), "selected_for_raw_branch"
            ] = 1
        else:
            selected = self.selector_table_.loc[
                self.selector_table_["selected_for_raw_branch"] == 1, "feature"
            ].tolist()
        self.selected_raw_features_ = selected
        self.selected_raw_idx_ = np.array([name_to_idx[f] for f in selected],
                                          dtype=int) if selected else np.array([], dtype=int)

        max_allowed = min(self.max_pca_components, X_proc.shape[0], X_proc.shape[1])
        probe = PCA(n_components=max_allowed, random_state=self.random_state)
        probe.fit(X_proc)
        cum = np.cumsum(probe.explained_variance_ratio_)
        n_by_var = int(np.searchsorted(cum, self.pca_variance_threshold) + 1)
        self.n_pca_components_ = max(2, min(max_allowed, n_by_var))
        self.pca_ = PCA(n_components=self.n_pca_components_, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_proc)

        if len(self.selected_raw_idx_):
            X_fused = np.hstack([X_pca, X_proc[:, self.selected_raw_idx_]])
        else:
            X_fused = X_pca.copy()

        self.lda_ = LinearDiscriminantAnalysis(solver="svd")
        self.lda_.fit(X_pca, y)

        self.logreg_ = _make_multinomial_logreg(class_weight="balanced")
        self.logreg_.fit(X_fused, y)

        Xb, yb = random_oversample(X_fused, y, random_state=self.random_state)
        self.mlp_ = _make_regularized_mlp(
            hidden_layer_sizes=self.mlp_hidden,
            random_state=self.random_state,
            max_iter=1600,
            n_iter_no_change=25,
        )
        self.mlp_.fit(Xb, yb)

        self.model_weights_ = self._auto_weights(X_pca, X_fused, y)
        self.model_weight_dict_ = {
            "LDA_PCA": float(self.model_weights_[0]),
            "LogReg_PCAplusRaw": float(self.model_weights_[1]),
            "MLP_PCAplusRaw": float(self.model_weights_[2]),
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        _, X_pca, X_fused = self._build_representation(X)
        p_lda = _align_proba(self.lda_.predict_proba(X_pca), self.lda_.classes_, self.classes_)
        p_log = _align_proba(self.logreg_.predict_proba(X_fused), self.logreg_.classes_, self.classes_)
        p_mlp = _align_proba(self.mlp_.predict_proba(X_fused), self.mlp_.classes_, self.classes_)
        probs = self.model_weights_[0] * p_lda + self.model_weights_[1] * p_log + self.model_weights_[2] * p_mlp
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return probs / row_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

def _extract_shap_matrix_for_class(shap_values, class_pos: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_pos])
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[2] > class_pos:
        return arr[:, :, class_pos]
    if arr.ndim == 3 and arr.shape[0] > class_pos:
        return arr[class_pos]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported SHAP value shape: {arr.shape}")

def _global_rank_fusion(
        feature_names: List[str],
        selector_df: pd.DataFrame,
        perm_df: pd.DataFrame,
        shap_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = selector_df[["feature", "avg_rank", "selected_for_raw_branch"]].copy()
    out = out.merge(perm_df[["feature", "perm_importance_mean"]], on="feature", how="left")
    if shap_df is not None:
        out = out.merge(shap_df[["feature", "mean_abs_shap_ckdu"]], on="feature", how="left")
    else:
        out["mean_abs_shap_ckdu"] = np.nan

    out["perm_importance_mean"] = out["perm_importance_mean"].fillna(0.0)
    out["mean_abs_shap_ckdu"] = out["mean_abs_shap_ckdu"].fillna(0.0)

    out["selector_rank_norm"] = 1.0 - (out["avg_rank"] - out["avg_rank"].min()) / max(
        out["avg_rank"].max() - out["avg_rank"].min(), 1e-12)
    out["perm_rank_norm"] = out["perm_importance_mean"].rank(ascending=False, method="average")
    out["perm_rank_norm"] = 1.0 - (out["perm_rank_norm"] - out["perm_rank_norm"].min()) / max(
        out["perm_rank_norm"].max() - out["perm_rank_norm"].min(), 1e-12)
    out["shap_rank_norm"] = out["mean_abs_shap_ckdu"].rank(ascending=False, method="average")
    out["shap_rank_norm"] = 1.0 - (out["shap_rank_norm"] - out["shap_rank_norm"].min()) / max(
        out["shap_rank_norm"].max() - out["shap_rank_norm"].min(), 1e-12)

    out["fusion_contribution_score"] = (
            0.25 * out["selector_rank_norm"] +
            0.35 * out["perm_rank_norm"] +
            0.35 * out["shap_rank_norm"] +
            0.05 * out["selected_for_raw_branch"]
    )
    out["legacy_alias"] = [REVERSE_F_ALIAS_MAP.get(f, "") for f in out["feature"]]
    return out.sort_values("fusion_contribution_score", ascending=False).reset_index(drop=True)

def analysis_fusion(
        ds: Dataset,
        outdir: str,
        n_splits: int = 5,
        preproc_kind: str = "robust",
        pca_variance_threshold: float = 0.95,
        max_pca_components: int = 12,
        raw_top_k: int = 8,
        forced_raw_features: Optional[List[str]] = None,
        shap_background: int = 40,
        shap_samples: int = 20,
        lime_samples: int = 5,
        perm_repeats: int = 10,
):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]
    class_array = np.asarray(labels_sorted)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred_oof = np.zeros_like(y)
    y_proba_oof = np.zeros((len(y), len(labels_sorted)), dtype=float)
    fold_metrics: List[Dict[str, float]] = []
    perm_rows: List[np.ndarray] = []
    selected_counts = {feat: 0 for feat in ds.feature_names}
    model_weight_rows = []

    scorer = make_scorer(f1_score, average="macro")

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = PCAGuidedFusionClassifier(
            preproc_kind=preproc_kind,
            pca_variance_threshold=pca_variance_threshold,
            max_pca_components=max_pca_components,
            raw_top_k=raw_top_k,
            forced_raw_features=forced_raw_features,
            internal_cv=3,
            random_state=RANDOM_STATE + fold,
            feature_names=ds.feature_names,
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])

        y_pred_oof[te] = pred
        y_proba_oof[te, :] = _align_proba(proba, model.classes_, class_array)

        m = _basic_metrics(y[te], pred)
        m["auc_macro_ovr"] = _safe_multiclass_auc(y[te], y_proba_oof[te, :])
        m["fold"] = fold
        fold_metrics.append(m)

        model_weight_rows.append({"fold": fold, **model.model_weight_dict_})

        for feat in model.selected_raw_features_:
            selected_counts[feat] += 1

        imp = permutation_importance(
            model,
            X[te],
            y[te],
            scoring=scorer,
            n_repeats=perm_repeats,
            random_state=RANDOM_STATE + fold,
            n_jobs=1,
        )
        perm_rows.append(imp.importances_mean.copy())

    metrics = _basic_metrics(y, y_pred_oof)
    metrics["auc_macro_ovr"] = _safe_multiclass_auc(y, y_proba_oof)
    _save_json(metrics, os.path.join(outdir, "fusion_cv_metrics.json"))

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(outdir, "fusion_cv_fold_metrics.csv"), index=False)

    weight_df = pd.DataFrame(model_weight_rows)
    weight_df.to_csv(os.path.join(outdir, "fusion_model_weights_by_fold.csv"), index=False)

    _save_confusion(
        y_true=y,
        y_pred=y_pred_oof,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "fusion_oof_confusion_matrix.png"),
        title=f"PCA-guided fusion model (OOF, {n_splits}-fold CV)",
    )

    with open(os.path.join(outdir, "fusion_oof_classification_report.txt"), "w") as f:
        f.write(classification_report(y, y_pred_oof, target_names=label_names, digits=3))

    perm_mat = np.vstack(perm_rows)
    perm_df = pd.DataFrame({
        "feature": ds.feature_names,
        "perm_importance_mean": perm_mat.mean(axis=0),
        "perm_importance_std": perm_mat.std(axis=0, ddof=1),
        "selected_raw_branch_count": [selected_counts[f] for f in ds.feature_names],
        "legacy_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in ds.feature_names],
    }).sort_values("perm_importance_mean", ascending=False)
    perm_df.to_csv(os.path.join(outdir, "fusion_permutation_importance.csv"), index=False)
    _top_bar_plot(
        perm_df,
        value_col="perm_importance_mean",
        title="Fusion model permutation importance (top 15)",
        outpath=os.path.join(outdir, "fusion_permutation_importance_top15.png"),
        top_n=15,
    )

    final_model = PCAGuidedFusionClassifier(
        preproc_kind=preproc_kind,
        pca_variance_threshold=pca_variance_threshold,
        max_pca_components=max_pca_components,
        raw_top_k=raw_top_k,
        forced_raw_features=forced_raw_features,
        internal_cv=3,
        random_state=RANDOM_STATE,
        feature_names=ds.feature_names,
    )
    final_model.fit(X, y)

    selector_df = final_model.selector_table_.copy()
    selector_df["legacy_alias"] = [REVERSE_F_ALIAS_MAP.get(f, "") for f in selector_df["feature"]]
    selector_df.to_csv(os.path.join(outdir, "fusion_raw_feature_selector_consensus.csv"), index=False)
    _save_json(
        {
            "selected_raw_features": final_model.selected_raw_features_,
            "selected_raw_features_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in
                                            final_model.selected_raw_features_],
            "forced_raw_features": forced_raw_features if forced_raw_features else [],
            "n_pca_components": int(final_model.n_pca_components_),
            "pca_variance_threshold": pca_variance_threshold,
            "model_weights": final_model.model_weight_dict_,
        },
        os.path.join(outdir, "fusion_final_model_summary.json"),
    )

    shap_df: Optional[pd.DataFrame] = None
    try:
        import shap  # type: ignore

        if 1 in final_model.classes_:
            ckdu_class_pos = int(np.where(final_model.classes_ == 1)[0][0])
        else:
            ckdu_class_pos = 0

        rng = np.random.default_rng(RANDOM_STATE)
        idx_ckdu = np.flatnonzero(y == 1) if 1 in np.unique(y) else np.arange(len(y))
        idx_all = np.arange(len(y))

        bg_size = min(shap_background, len(idx_all))
        ex_size = min(shap_samples, len(idx_ckdu) if len(idx_ckdu) else len(idx_all))
        bg_idx = rng.choice(idx_all, size=bg_size, replace=False)
        if len(idx_ckdu):
            explain_idx = rng.choice(idx_ckdu, size=ex_size, replace=False)
        else:
            explain_idx = rng.choice(idx_all, size=ex_size, replace=False)

        background = X[bg_idx]
        explain_X = X[explain_idx]

        explainer = shap.KernelExplainer(final_model.predict_proba, background)
        shap_values = explainer.shap_values(explain_X, nsamples="auto")
        ckdu_shap = _extract_shap_matrix_for_class(shap_values, ckdu_class_pos)

        shap_df = pd.DataFrame({
            "feature": ds.feature_names,
            "mean_abs_shap_ckdu": np.mean(np.abs(ckdu_shap), axis=0),
            "mean_signed_shap_ckdu": np.mean(ckdu_shap, axis=0),
            "legacy_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in ds.feature_names],
        }).sort_values("mean_abs_shap_ckdu", ascending=False)
        shap_df.to_csv(os.path.join(outdir, "fusion_shap_global_ckdu.csv"), index=False)
        _top_bar_plot(
            shap_df,
            value_col="mean_abs_shap_ckdu",
            title="Fusion model SHAP importance for CKDu (top 15)",
            outpath=os.path.join(outdir, "fusion_shap_global_ckdu_top15.png"),
            top_n=15,
        )

        per_sample_df = pd.DataFrame(ckdu_shap, columns=ds.feature_names)
        per_sample_df.insert(0, "sample_index", explain_idx)
        per_sample_df.to_csv(os.path.join(outdir, "fusion_shap_ckdu_per_sample.csv"), index=False)
    except Exception as e:
        with open(os.path.join(outdir, "fusion_shap_skipped.txt"), "w") as f:
            f.write(f"SHAP could not be generated: {e}\n")

    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore

        class_names = [LABEL_MAP[int(c)] for c in final_model.classes_]
        lime_outdir = _ensure_outdir(os.path.join(outdir, "lime_explanations"))
        explainer = LimeTabularExplainer(
            training_data=X,
            feature_names=ds.feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=RANDOM_STATE,
        )

        if 1 in np.unique(y):
            idx_pool = np.flatnonzero(y == 1)
        else:
            idx_pool = np.arange(len(y))
        rng = np.random.default_rng(RANDOM_STATE)
        chosen = rng.choice(idx_pool, size=min(lime_samples, len(idx_pool)), replace=False)
        ckdu_pos = int(np.where(final_model.classes_ == 1)[0][0]) if 1 in final_model.classes_ else 0
        lime_rows = []
        for idx in chosen:
            exp = explainer.explain_instance(
                data_row=X[idx],
                predict_fn=final_model.predict_proba,
                labels=[ckdu_pos],
                num_features=min(12, X.shape[1]),
            )
            html_path = os.path.join(lime_outdir, f"lime_sample_{idx}_ckdu.html")
            txt_path = os.path.join(lime_outdir, f"lime_sample_{idx}_ckdu.txt")
            exp.save_to_file(html_path)
            items = exp.as_list(label=ckdu_pos)
            with open(txt_path, "w") as f:
                for cond, weight in items:
                    f.write(f"{cond}\t{weight}\n")
            for cond, weight in items:
                lime_rows.append({"sample_index": int(idx), "rule": cond, "weight": float(weight)})
        pd.DataFrame(lime_rows).to_csv(os.path.join(outdir, "fusion_lime_summary_ckdu.csv"), index=False)
    except Exception as e:
        with open(os.path.join(outdir, "fusion_lime_skipped.txt"), "w") as f:
            f.write(
                "LIME explanations were skipped. Install the package with: pip install lime\n"
                f"Underlying error: {e}\n"
            )

    fused_rank_df = _global_rank_fusion(
        feature_names=ds.feature_names,
        selector_df=selector_df,
        perm_df=perm_df,
        shap_df=shap_df,
    )
    fused_rank_df.to_csv(os.path.join(outdir, "fusion_final_element_ranking.csv"), index=False)
    _top_bar_plot(
        fused_rank_df,
        value_col="fusion_contribution_score",
        title="Final fusion-based element ranking (top 15)",
        outpath=os.path.join(outdir, "fusion_final_element_ranking_top15.png"),
        top_n=15,
    )

    print("\n[Fusion] CV metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n[Fusion] Selected raw-feature branch:")
    print(final_model.selected_raw_features_)
    print("\n[Fusion] Final top-ranked elements:")
    print(fused_rank_df.head(12).to_string(index=False))
    print(f"\n[Fusion] Saved outputs to: {outdir}")

# ==================
# 11) Run everything
# ==================
def analysis_all(ds: Dataset, outdir: str, args):
    root = _ensure_outdir(outdir)
    analysis_pca(ds, os.path.join(root, "pca"))
    analysis_skew_kurt(ds, os.path.join(root, "skew_kurt"))
    analysis_significance(ds, os.path.join(root, "significance"))
    analysis_bayes(ds, os.path.join(root, "bayes"), n_splits=args.cv_splits)
    analysis_fisher(ds, os.path.join(root, "fisher"), n_splits=args.cv_splits)
    analysis_regression(ds, os.path.join(root, "regression"))
    analysis_gmm(ds, os.path.join(root, "gmm"))
    analysis_nn(ds, os.path.join(root, "nn"))

    consensus_top, consensus_summary_df, consensus_hits_df = _derive_frequency_consensus_features_from_outputs(
        root_outdir=root,
        feature_names=ds.feature_names,
        top_k=args.consensus_top_k,
        source_top_n=args.consensus_source_top_n,
    )
    if consensus_top:
        selected_top_features = consensus_top
        selection_mode = "frequency_consensus_from_previous_analyses"
    else:
        selected_top_features = [s.strip() for s in args.top_features.split(",") if s.strip()]
        selected_top_features = resolve_feature_names(selected_top_features, ds.feature_names)
        selection_mode = "fallback_manual_top_features"

    consensus_summary_df.to_csv(os.path.join(root, "consensus_feature_frequency_summary.csv"), index=False)
    if not consensus_hits_df.empty:
        consensus_hits_df.to_csv(os.path.join(root, "consensus_feature_frequency_hits.csv"), index=False)
    _save_json(
        {
            "selection_mode": selection_mode,
            "consensus_top_features": selected_top_features,
            "consensus_top_features_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in selected_top_features],
            "consensus_top_k": int(args.consensus_top_k),
            "consensus_source_top_n": int(args.consensus_source_top_n),
        },
        os.path.join(root, "consensus_feature_selection.json"),
    )
    print("\n[All] Consensus top features from previous analyses:")
    print(selected_top_features)

    analysis_validate_top_features(
        ds,
        outdir=os.path.join(root, "validate_top8"),
        top_features=selected_top_features,
        preproc_kind=args.preproc,
        n_splits=args.cv_splits,
        perm_repeats=args.perm_repeats,
    )
    analysis_fusion(
        ds,
        outdir=os.path.join(root, "fusion"),
        n_splits=args.cv_splits,
        preproc_kind=args.preproc,
        pca_variance_threshold=args.fusion_pca_var,
        max_pca_components=args.fusion_max_pca,
        raw_top_k=max(args.fusion_raw_top_k, len(selected_top_features)),
        forced_raw_features=selected_top_features,
        shap_background=args.shap_background,
        shap_samples=args.shap_samples,
        lime_samples=args.lime_samples,
        perm_repeats=args.perm_repeats,
    )

# =====
# Main
# =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pattern/presentation/CKDu_processed.csv", help="Path to CKDu_processed.csv")
    ap.add_argument(
        "--analysis",
        required=True,
        choices=[
            "pca",
            "bayes",
            "fisher",
            "skew_kurt",
            "significance",
            "regression",
            "gmm",
            "nn",
            "validate_top8",
            "fusion",
            "all",
        ],
        help="Which analysis to run",
    )
    ap.add_argument("--outdir", default="pattern/presentation/ckdu_results", help="Output directory")

    ap.add_argument(
        "--top_features",
        default="Ag,Ca,Na,K,Al,V,Sr,In",
        help=(
            "Comma-separated consensus features to validate. "
            "You may use real names (e.g. Ag,Ca,Na) or legacy aliases (e.g. F21,F4,F1)."
        ),
    )
    ap.add_argument(
        "--preproc",
        default="robust",
        choices=["robust", "standard"],
        help="Preprocessing pipeline for validate_top8/fusion",
    )
    ap.add_argument("--cv_splits", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--perm_repeats", type=int, default=10, help="Permutation repeats")
    ap.add_argument(
        "--consensus_top_k",
        type=int,
        default=8,
        help="When --analysis all: number of frequency-consensus features to use for validate_top8 and fusion",
    )
    ap.add_argument(
        "--consensus_source_top_n",
        type=int,
        default=8,
        help="When --analysis all: top-N features taken from each prior analysis output for consensus counting",
    )

    ap.add_argument("--fusion_raw_top_k", type=int, default=8,
                    help="Number of raw features in the fusion side-branch")
    ap.add_argument("--fusion_max_pca", type=int, default=12,
                    help="Maximum PCA components for the fusion branch")
    ap.add_argument("--fusion_pca_var", type=float, default=0.95,
                    help="Cumulative variance target for fusion PCA")
    ap.add_argument("--shap_background", type=int, default=40, help="Background samples for SHAP")
    ap.add_argument("--shap_samples", type=int, default=20, help="Number of samples to explain with SHAP")
    ap.add_argument("--lime_samples", type=int, default=5, help="Number of local CKDu samples for LIME")

    args = ap.parse_args()

    ds = load_ckdu_processed(args.csv)

    if args.analysis == "all":
        analysis_all(ds, outdir=args.outdir, args=args)
        return

    outdir = os.path.join(args.outdir, args.analysis)

    if args.analysis == "pca":
        analysis_pca(ds, outdir=outdir)
    elif args.analysis == "bayes":
        analysis_bayes(ds, outdir=outdir, n_splits=args.cv_splits)
    elif args.analysis == "fisher":
        analysis_fisher(ds, outdir=outdir, n_splits=args.cv_splits)
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
    elif args.analysis == "validate_top8":
        top_feats = [s.strip() for s in args.top_features.split(",") if s.strip()]
        analysis_validate_top_features(
            ds,
            outdir=outdir,
            top_features=top_feats,
            preproc_kind=args.preproc,
            n_splits=args.cv_splits,
            perm_repeats=args.perm_repeats,
        )
    elif args.analysis == "fusion":
        analysis_fusion(
            ds,
            outdir=outdir,
            n_splits=args.cv_splits,
            preproc_kind=args.preproc,
            pca_variance_threshold=args.fusion_pca_var,
            max_pca_components=args.fusion_max_pca,
            raw_top_k=args.fusion_raw_top_k,
            shap_background=args.shap_background,
            shap_samples=args.shap_samples,
            lime_samples=args.lime_samples,
            perm_repeats=args.perm_repeats,
        )
    else:
        raise ValueError("Unknown analysis")

if __name__ == "__main__":
    main()
