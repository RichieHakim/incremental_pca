"""
Diagnostic script to understand the subspace angle issue.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA, IncrementalPCA as SklearnIPCA
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from incremental_pca_torch import IncrementalPCA


def get_subspace_angle_v2(A: np.ndarray, B: np.ndarray) -> float:
    """Alternative subspace angle using orthogonalization."""
    # QR decomposition to orthonormalize
    Q_A, _ = np.linalg.qr(A.T)  # (n_features, n_components)
    Q_B, _ = np.linalg.qr(B.T)  # (n_features, n_components)
    
    # SVD of Q_A.T @ Q_B
    _, s, _ = np.linalg.svd(Q_A.T @ Q_B)
    s = np.clip(s, -1, 1)
    angles = np.arccos(s)
    return np.degrees(np.max(angles))


def check_orthonormality(V):
    """Check if rows are orthonormal."""
    VVT = V @ V.T
    identity = np.eye(V.shape[0])
    error = np.max(np.abs(VVT - identity))
    return error


np.random.seed(42)
torch.manual_seed(42)

n_samples, n_features = 1000, 100
n_components = 10
batch_size = 100

X = np.random.randn(n_samples, n_features).astype(np.float64)

# Reference: sklearn batch PCA
sklearn_pca = PCA(n_components=n_components)
sklearn_pca.fit(X)
sklearn_components = sklearn_pca.components_

# Our implementation - full batch
ipca_full = IncrementalPCA(
    n_components=n_components,
    batch_size=n_samples,
    device='cpu',
    dtype=torch.float64,
)
ipca_full.fit(X)
our_full_components = ipca_full.components_.cpu().numpy()

# Our implementation - incremental
ipca_inc = IncrementalPCA(
    n_components=n_components,
    batch_size=batch_size,
    device='cpu',
    dtype=torch.float64,
)
ipca_inc.fit(X)
our_inc_components = ipca_inc.components_.cpu().numpy()

# sklearn IncrementalPCA
sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
sklearn_ipca.fit(X)
sklearn_ipca_components = sklearn_ipca.components_

print("=" * 60)
print("Orthonormality Check")
print("=" * 60)
print(f"sklearn PCA orthonormality error: {check_orthonormality(sklearn_components):.2e}")
print(f"Our full-batch orthonormality error: {check_orthonormality(our_full_components):.2e}")
print(f"Our incremental orthonormality error: {check_orthonormality(our_inc_components):.2e}")
print(f"sklearn IncrementalPCA orthonormality error: {check_orthonormality(sklearn_ipca_components):.2e}")

print("\n" + "=" * 60)
print("Subspace Angles (using QR-based method)")
print("=" * 60)
print(f"Our full-batch vs sklearn PCA: {get_subspace_angle_v2(our_full_components, sklearn_components):.2f}°")
print(f"Our incremental vs sklearn PCA: {get_subspace_angle_v2(our_inc_components, sklearn_components):.2f}°")
print(f"sklearn IPCA vs sklearn PCA: {get_subspace_angle_v2(sklearn_ipca_components, sklearn_components):.2f}°")
print(f"Our incremental vs sklearn IPCA: {get_subspace_angle_v2(our_inc_components, sklearn_ipca_components):.2f}°")
print(f"Our full-batch vs Our incremental: {get_subspace_angle_v2(our_full_components, our_inc_components):.2f}°")

print("\n" + "=" * 60)
print("Explained Variance Ratios (sum)")
print("=" * 60)
print(f"sklearn PCA: {sklearn_pca.explained_variance_ratio_.sum():.4f}")
print(f"Our full-batch: {ipca_full.explained_variance_ratio_.sum().item():.4f}")
print(f"Our incremental: {ipca_inc.explained_variance_ratio_.sum().item():.4f}")
print(f"sklearn IPCA: {sklearn_ipca.explained_variance_ratio_.sum():.4f}")

print("\n" + "=" * 60)
print("Reconstruction Error")
print("=" * 60)

def reconstruction_error(pca, X):
    if hasattr(pca, 'inverse_transform'):
        if hasattr(pca.transform(X[:1]), 'numpy'):
            X_rec = pca.inverse_transform(pca.transform(X)).numpy()
        else:
            X_rec = pca.inverse_transform(pca.transform(X))
        return np.mean((X - X_rec) ** 2)
    return None

print(f"sklearn PCA: {reconstruction_error(sklearn_pca, X):.6f}")
print(f"Our full-batch: {reconstruction_error(ipca_full, X):.6f}")
print(f"Our incremental: {reconstruction_error(ipca_inc, X):.6f}")
print(f"sklearn IPCA: {reconstruction_error(sklearn_ipca, X):.6f}")

print("\n" + "=" * 60)
print("First 3 Components (first 5 elements) - comparing sklearn vs our full-batch")
print("=" * 60)
print("sklearn PCA components:")
print(sklearn_components[:3, :5])
print("\nOur full-batch components:")
print(our_full_components[:3, :5])
print("\nSign-aligned difference (should be small):")
for i in range(3):
    sign = np.sign(np.dot(our_full_components[i], sklearn_components[i]))
    diff = our_full_components[i] * sign - sklearn_components[i]
    print(f"  Component {i}: max diff = {np.max(np.abs(diff)):.2e}")
