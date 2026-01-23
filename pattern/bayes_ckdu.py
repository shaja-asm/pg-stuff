import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)

PATH = r"C:\Users\shaja\Downloads\CKDu.csv"
LABEL_COL = "Group.1"

MINERALS = ['Na','Mg','K','Ca','Al','V','Ti','Cr','Mn','Fe','Co','Ni','Cu','Zn','Rb','Sr','Ba','Pb']

N_SPLITS = 5
RANDOM_STATE = 42
TOPK = 10

def cv_scores(model, Xmat, yvec, folds, threshold=0.5):
    """Return per-fold AUC/ACC plus out-of-fold probabilities & labels."""
    aucs, accs = [], []
    oof_p = np.zeros(len(yvec), dtype=float)
    oof_pred = np.zeros(len(yvec), dtype=int)

    for tr, te in folds:
        model.fit(Xmat[tr], yvec[tr])
        p = model.predict_proba(Xmat[te])[:, 1]
        pred = (p >= threshold).astype(int)

        oof_p[te] = p
        oof_pred[te] = pred

        aucs.append(roc_auc_score(yvec[te], p))
        accs.append(accuracy_score(yvec[te], pred))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "oof_p": oof_p,
        "oof_pred": oof_pred,
        "aucs": aucs,
        "accs": accs,
    }

def plot_mean_roc(y_true, y_score, folds, title="ROC (k-fold CV)"):
    """Plot per-fold ROC + mean ROC with std band."""
    mean_fpr = np.linspace(0, 1, 400)
    tprs, aucs = [], []

    plt.figure()
    for i, (_, te) in enumerate(folds, start=1):
        fpr, tpr, _ = roc_curve(y_true[te], y_score[te])
        auc = roc_auc_score(y_true[te], y_score[te])
        aucs.append(auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        plt.plot(fpr, tpr, linewidth=1, label=f"Fold {i} (AUC={auc:.3f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0

    std_tpr = np.std(tprs, axis=0, ddof=1) if len(tprs) > 1 else np.zeros_like(mean_tpr)

    plt.plot(mean_fpr, mean_tpr, linewidth=2, label=f"Mean (AUC={mean_auc:.3f}±{std_auc:.3f})")
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.2,
        label="±1 std. dev."
    )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

def log_gauss_pdf(x, mean, var):
    """Vectorized log N(x; mean, var) for each feature."""
    return -0.5*np.log(2*np.pi*var) - 0.5*((x - mean)**2)/var

def bayes_llr_contributions(X, y, minerals, topk=10):
    """
    Compute per-mineral log-likelihood ratio (LLR) contributions:
        llr_ij = log p(x_ij|CKDu) - log p(x_ij|non-CKDu)
    """
    nb = GaussianNB()
    nb.fit(X, y)
    mu = nb.theta_
    var = nb.var_

    llr = log_gauss_pdf(X, mu[1], var[1]) - log_gauss_pdf(X, mu[0], var[0])

    avg_abs_llr = np.mean(np.abs(llr), axis=0)
    avg_llr = np.mean(llr, axis=0)

    evidence_df = pd.DataFrame({
        "Mineral": minerals,
        "AvgAbsLLR": avg_abs_llr,
        "AvgLLR": avg_llr
    }).sort_values("AvgAbsLLR", ascending=False)

    top = evidence_df.head(topk).iloc[::-1]

    plt.figure()
    plt.barh(top["Mineral"], top["AvgAbsLLR"])
    plt.xlabel("Average |log-likelihood ratio| contribution")
    plt.title(f"Top {topk} Minerals by Bayes Decision-Rule Contribution (LLR magnitude)")
    plt.tight_layout()

    plt.figure()
    plt.barh(top["Mineral"], top["AvgLLR"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Average log-likelihood ratio (signed)")
    plt.title("Direction of Evidence: + favors CKDu, − favors non-CKDu")
    plt.tight_layout()

    return evidence_df

def boxplots_for_top_minerals(X, y, minerals, top_feats, title="Boxplots: CKDu vs non-CKDu"):
    """
    Create grouped boxplots (non-CKDu vs CKDu) for each mineral in top_feats.
    Uses matplotlib only (no seaborn).
    """
    feat_idx = [minerals.index(f) for f in top_feats]

    non = X[y == 0][:, feat_idx]
    ckdu = X[y == 1][:, feat_idx]

    # Build alternating list: [non for feat1, ckdu for feat1, non for feat2, ckdu for feat2, ...]
    data = []
    positions = []
    labels = []
    pos = 1

    for i, f in enumerate(top_feats):
        data.append(non[:, i])
        data.append(ckdu[:, i])
        positions.extend([pos, pos + 1])
        labels.extend([f"{f}\nnon", f"{f}\nCKDu"])
        pos += 3  # gap between minerals

    plt.figure(figsize=(max(10, len(top_feats)*1.3), 5))
    plt.boxplot(data, positions=positions, widths=0.7, showfliers=True)
    plt.xticks(positions, labels, rotation=0)
    plt.ylabel("Mineral level")
    plt.title(title)
    plt.tight_layout()

def main():
    # LOAD + PREP
    df = pd.read_csv(PATH)
    df = df.dropna(subset=[LABEL_COL]).copy()

    # numeric cast
    for c in MINERALS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # binary target: CKDu vs non-CKDu
    y = (df[LABEL_COL].astype(str).str.strip() == "CKDu").astype(int).values
    X = df[MINERALS].values

    # impute
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    # priors (from data)
    p1 = y.mean()
    p0 = 1 - p1
    print(f"P(CKDu)={p1:.4f}, P(non-CKDu)={p0:.4f}  |  N={len(y)} (CKDu={y.sum()}, non={len(y)-y.sum()})")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))

    # FULL BAYES MODEL (ALL MINERALS)
    full_model = GaussianNB()
    full = cv_scores(full_model, X, y, folds, threshold=0.5)
    print(f"\nFull model: AUC={full['auc_mean']:.3f}±{full['auc_std']:.3f}, "
          f"Acc={full['acc_mean']:.3f}±{full['acc_std']:.3f}")

    plot_mean_roc(y, full["oof_p"], folds, title="Gaussian Naïve Bayes ROC (CKDu vs non-CKDu)")

    cm = confusion_matrix(y, full["oof_pred"], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-CKDu", "CKDu"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format='d')
    plt.title("Confusion Matrix (OOF predictions, threshold=0.5)")
    plt.tight_layout()

    # SINGLE-MINERAL BAYES RANKING (CV AUC)
    single_rows = []
    for j, feat in enumerate(MINERALS):
        res = cv_scores(GaussianNB(), X[:, [j]], y, folds, threshold=0.5)
        single_rows.append((feat, res["auc_mean"], res["acc_mean"]))

    single_df = pd.DataFrame(
        sorted(single_rows, key=lambda t: t[1], reverse=True),
        columns=["Mineral", "CV_AUC", "CV_Acc"]
    )

    print("\nTop single-mineral predictors (by CV AUC):")
    print(single_df.head(TOPK).to_string(index=False))

    plt.figure()
    plt.barh(single_df["Mineral"].head(TOPK)[::-1], single_df["CV_AUC"].head(TOPK)[::-1])
    plt.xlabel("CV AUC (single mineral)")
    plt.title(f"Top {TOPK} Single-Mineral Bayes Predictors (AUC)")
    plt.tight_layout()

    # LEAVE-ONE-OUT AUC DROP
    base_auc = full["auc_mean"]
    drop_rows = []
    for j, feat in enumerate(MINERALS):
        cols = [i for i in range(len(MINERALS)) if i != j]
        res = cv_scores(GaussianNB(), X[:, cols], y, folds, threshold=0.5)
        drop_rows.append((feat, base_auc - res["auc_mean"]))

    drop_df = pd.DataFrame(
        sorted(drop_rows, key=lambda t: t[1], reverse=True),
        columns=["Mineral", "AUC_Drop_When_Removed"]
    )

    print("\nTop minerals by AUC drop when removed (bigger = more important in full model):")
    print(drop_df.head(TOPK).to_string(index=False))

    plt.figure()
    plt.barh(drop_df["Mineral"].head(TOPK)[::-1], drop_df["AUC_Drop_When_Removed"].head(TOPK)[::-1])
    plt.xlabel("AUC drop (full - without mineral)")
    plt.title(f"Top {TOPK} Minerals by Leave-One-Out AUC Drop")
    plt.tight_layout()

    # MEAN ± SD COMPARISON (Top 8 by AUC drop)
    top_feats = drop_df["Mineral"].head(8).tolist()
    idx = [MINERALS.index(f) for f in top_feats]

    X_ckdu = X[y == 1][:, idx]
    X_non  = X[y == 0][:, idx]

    mean_ckdu = X_ckdu.mean(axis=0)
    mean_non  = X_non.mean(axis=0)
    std_ckdu  = X_ckdu.std(axis=0, ddof=1)
    std_non   = X_non.std(axis=0, ddof=1)

    xpos = np.arange(len(top_feats))

    plt.figure(figsize=(9, 4.8))
    plt.errorbar(xpos - 0.05, mean_non, yerr=std_non, fmt='o', capsize=3, label="non-CKDu")
    plt.errorbar(xpos + 0.05, mean_ckdu, yerr=std_ckdu, fmt='o', capsize=3, label="CKDu")
    plt.xticks(xpos, top_feats, rotation=30, ha="right")
    plt.ylabel("Mineral level (mean ± SD)")
    plt.title("Top Minerals: CKDu vs non-CKDu (mean ± SD)")
    plt.legend()
    plt.tight_layout()

    # LLR CONTRIBUTIONS (inside the Bayes rule)
    evidence_df = bayes_llr_contributions(X, y, MINERALS, topk=TOPK)
    print("\nTop minerals by Bayes decision-rule contribution (AvgAbsLLR):")
    print(evidence_df.head(TOPK).to_string(index=False))

    # Box plots(Top minerals)
    # Box plots for top 8 by AUC drop (matches your mean±SD chart)
    boxplots_for_top_minerals(
        X, y, MINERALS, top_feats,
        title="Boxplots (Top 8 by AUC-drop): non-CKDu vs CKDu"
    )

    # Box plots for top 8 by LLR magnitude
    top_llr_feats = evidence_df["Mineral"].head(8).tolist()
    boxplots_for_top_minerals(
        X, y, MINERALS, top_llr_feats,
        title="Boxplots (Top 8 by LLR): non-CKDu vs CKDu"
    )

    plt.show()

if __name__ == "__main__":
    main()
