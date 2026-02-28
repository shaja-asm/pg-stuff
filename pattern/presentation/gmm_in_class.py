import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# ==========================
# 1. Load and preprocess data
# ==========================
iris = load_iris()
X = iris.data
y_true = iris.target  # true species (for comparison only)

# Standardize features (very important for GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# 2. Find optimal number of components using BIC & AIC
# ==========================
n_components_range = range(1, 11)
bic_scores = []
aic_scores = []
models = []

print("Fitting GMM for different number of components...\n")

for n in n_components_range:
    gmm = GaussianMixture(
        n_components=n,
        covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
        random_state=42,
        n_init=10  # multiple initializations for stability
    )
    gmm.fit(X_scaled)

    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))
    models.append(gmm)

    print(f"n_components = {n:2d} | BIC = {gmm.bic(X_scaled):.1f} | AIC = {gmm.aic(X_scaled):.1f}")

# Find best model according to BIC (preferred for GMM)
best_n_bic = n_components_range[np.argmin(bic_scores)]
best_n_aic = n_components_range[np.argmin(aic_scores)]

print(f"\nBest number of Gaussians (by BIC): {best_n_bic}")
print(f"Best number of Gaussians (by AIC): {best_n_aic}")

# ==========================
# 3. Plot BIC and AIC curves
# ==========================
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, 'o-', label='BIC', color='blue')
plt.plot(n_components_range, aic_scores, 's-', label='AIC', color='red')
plt.axvline(x=best_n_bic, color='blue', linestyle='--', alpha=0.7, label=f'Best BIC: {best_n_bic}')
plt.xlabel('Number of Components (Gaussians)')
plt.ylabel('Information Criterion Score')
plt.title('Model Selection for GMM on Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 4. Fit the best model and visualize clusters
# ==========================
best_gmm = GaussianMixture(
    n_components=best_n_bic,
    covariance_type='full',
    random_state=42,
    n_init=10
)
best_gmm.fit(X_scaled)

# Predict cluster labels
y_pred = best_gmm.predict(X_scaled)

# Project to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=y_pred, cmap='viridis', s=60, edgecolor='k', alpha=0.8)
plt.title(f'GMM Clustering (n_components = {best_n_bic}) on Iris Dataset (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Predicted Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# ==========================
# 5. Evaluation
# ==========================
ari = adjusted_rand_score(y_true, y_pred)
print(f"\nAdjusted Rand Index (vs true species): {ari:.4f}")
print("ARI close to 1.0 means excellent clustering performance.")