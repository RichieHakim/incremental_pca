"""
Accuracy tests for Incremental PCA.

These tests compare the torch implementation against scikit-learn's
PCA and IncrementalPCA implementations. Due to floating-point differences
and slight algorithmic variations, we use tolerances rather than exact
equality.

Key differences to account for:
1. Sign ambiguity: PCA components can be flipped (negated) and still be valid.
2. Numerical precision: torch and numpy may produce slightly different results.
3. Algorithm variations: Incremental PCA updates differ slightly between implementations.
"""

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA, IncrementalPCA as SklearnIPCA

import sys
from pathlib import Path

# Add src to path for testing without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from incremental_pca_torch import IncrementalPCA


# =============================================================================
# Helper Functions
# =============================================================================

def align_signs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Align signs of rows in A to match B.
    
    PCA components are only unique up to sign. This function flips rows of A
    so that they have the same sign orientation as corresponding rows in B.
    
    Args:
        A: Array of shape (n_components, n_features)
        B: Reference array of same shape
        
    Returns:
        A with rows potentially sign-flipped to match B
    """
    # Compute dot product between corresponding rows
    dots = np.sum(A * B, axis=1)
    signs = np.sign(dots)
    # Handle zero dot products (shouldn't happen with real data)
    signs[signs == 0] = 1
    return A * signs[:, np.newaxis]


def components_close(
    A: np.ndarray, 
    B: np.ndarray, 
    rtol: float = 1e-4, 
    atol: float = 1e-6
) -> bool:
    """
    Check if two component matrices are close, accounting for sign ambiguity.
    """
    A_aligned = align_signs(A, B)
    return np.allclose(A_aligned, B, rtol=rtol, atol=atol)


def get_subspace_angle(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the largest principal angle between two subspaces.
    
    Uses the method based on computing singular values of A @ B.T where
    A and B have shape (n_components, n_features) with orthonormal rows.
    
    Returns angle in degrees. Close to 0 means subspaces are well-aligned.
    """
    # Ensure we're working with normalized rows
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    
    # Compute A @ B.T: shape (n_components, n_components)
    # The singular values of this matrix are cosines of principal angles
    M = A_norm @ B_norm.T
    _, s, _ = np.linalg.svd(M)
    
    # Clamp to valid range for arccos (numerical precision issues)
    s = np.clip(s, -1, 1)
    
    # Principal angles are arccos of singular values
    angles = np.arccos(s)
    
    # Return the largest angle (smallest singular value corresponds to largest angle)
    return np.degrees(np.max(angles))


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestBasicFunctionality:
    """Tests for basic model operations."""
    
    def test_fit_returns_self(self, small_data, device):
        """Fit should return the model instance."""
        ipca = IncrementalPCA(n_components=5, device=device)
        result = ipca.fit(small_data)
        assert result is ipca
    
    def test_attributes_after_fit(self, small_data, device):
        """Check that all expected attributes are set after fitting."""
        ipca = IncrementalPCA(n_components=5, device=device)
        ipca.fit(small_data)
        
        assert ipca.components_ is not None
        assert ipca.mean_ is not None
        assert ipca.var_ is not None
        assert ipca.singular_values_ is not None
        assert ipca.explained_variance_ is not None
        assert ipca.explained_variance_ratio_ is not None
        assert ipca.noise_variance_ is not None
        assert ipca.n_samples_seen_ == small_data.shape[0]
        assert ipca.n_components_ == 5
    
    def test_component_shapes(self, small_data, device):
        """Check output shapes are correct."""
        n_samples, n_features = small_data.shape
        n_components = 10
        
        ipca = IncrementalPCA(n_components=n_components, device=device)
        ipca.fit(small_data)
        
        assert ipca.components_.shape == (n_components, n_features)
        assert ipca.mean_.shape == (n_features,)
        assert ipca.var_.shape == (n_features,)
        assert ipca.singular_values_.shape == (n_components,)
        assert ipca.explained_variance_.shape == (n_components,)
        assert ipca.explained_variance_ratio_.shape == (n_components,)
    
    def test_transform_shape(self, small_data, device):
        """Check transform output shape."""
        n_components = 10
        ipca = IncrementalPCA(n_components=n_components, device=device)
        ipca.fit(small_data)
        
        X_transformed = ipca.transform(small_data)
        assert X_transformed.shape == (small_data.shape[0], n_components)
    
    def test_transform_without_fit_raises(self, small_data, device):
        """Transform without fitting should raise RuntimeError."""
        ipca = IncrementalPCA(n_components=5, device=device)
        with pytest.raises(RuntimeError):
            ipca.transform(small_data)
    
    def test_inverse_transform_shape(self, small_data, device):
        """Check inverse_transform output shape."""
        n_components = 10
        ipca = IncrementalPCA(n_components=n_components, device=device)
        ipca.fit(small_data)
        
        X_transformed = ipca.transform(small_data)
        X_reconstructed = ipca.inverse_transform(X_transformed)
        
        assert X_reconstructed.shape == small_data.shape
    
    def test_fit_transform_equals_fit_then_transform(self, small_data, device):
        """fit_transform should give same result as fit followed by transform."""
        ipca1 = IncrementalPCA(n_components=10, batch_size=50, device=device)
        X1 = ipca1.fit_transform(small_data)
        
        ipca2 = IncrementalPCA(n_components=10, batch_size=50, device=device)
        ipca2.fit(small_data)
        X2 = ipca2.transform(small_data)
        
        assert torch.allclose(X1, X2)


# =============================================================================
# Comparison with sklearn PCA (Full-Batch Reference)
# =============================================================================

class TestComparisonWithSklearnPCA:
    """
    Compare against sklearn's batch PCA as the gold standard.
    
    When batch_size >= n_samples, our implementation should closely
    match sklearn's PCA (modulo numerical differences).
    """
    
    def test_components_match_sklearn_pca(self, medium_data, device):
        """Components should match sklearn PCA when fit on full data."""
        n_components = 20
        
        # sklearn PCA (reference)
        sklearn_pca = PCA(n_components=n_components)
        sklearn_pca.fit(medium_data)
        
        # Our implementation with large batch (full data in one batch)
        ipca = IncrementalPCA(
            n_components=n_components, 
            batch_size=len(medium_data),
            device=device,
            dtype=torch.float64,  # Match sklearn precision
        )
        ipca.fit(medium_data)
        
        our_components = ipca.components_.cpu().numpy()
        sklearn_components = sklearn_pca.components_
        
        # Components should be close (up to sign)
        assert components_close(our_components, sklearn_components, rtol=1e-4)
    
    def test_explained_variance_matches_sklearn_pca(self, medium_data, device):
        """Explained variance should match sklearn PCA."""
        n_components = 20
        
        sklearn_pca = PCA(n_components=n_components)
        sklearn_pca.fit(medium_data)
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=len(medium_data),
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        our_var = ipca.explained_variance_.cpu().numpy()
        sklearn_var = sklearn_pca.explained_variance_
        
        assert np.allclose(our_var, sklearn_var, rtol=1e-3)
    
    def test_explained_variance_ratio_matches_sklearn_pca(self, medium_data, device):
        """Explained variance ratio should match sklearn PCA."""
        n_components = 20
        
        sklearn_pca = PCA(n_components=n_components)
        sklearn_pca.fit(medium_data)
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=len(medium_data),
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        our_ratio = ipca.explained_variance_ratio_.cpu().numpy()
        sklearn_ratio = sklearn_pca.explained_variance_ratio_
        
        assert np.allclose(our_ratio, sklearn_ratio, rtol=1e-3)
    
    def test_transform_matches_sklearn_pca(self, medium_data, device):
        """Transformed data should match sklearn PCA."""
        n_components = 20
        
        sklearn_pca = PCA(n_components=n_components)
        sklearn_pca.fit(medium_data)
        sklearn_transformed = sklearn_pca.transform(medium_data)
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=len(medium_data),
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        our_transformed = ipca.transform(medium_data).numpy()
        
        # Align signs before comparison
        our_transformed_aligned = our_transformed.copy()
        for i in range(n_components):
            if np.dot(our_transformed[:, i], sklearn_transformed[:, i]) < 0:
                our_transformed_aligned[:, i] *= -1
        
        assert np.allclose(our_transformed_aligned, sklearn_transformed, rtol=1e-4)
    
    def test_mean_matches_sklearn_pca(self, medium_data, device):
        """Mean should exactly match sklearn PCA."""
        ipca = IncrementalPCA(
            n_components=20,
            batch_size=len(medium_data),
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        sklearn_pca = PCA(n_components=20)
        sklearn_pca.fit(medium_data)
        
        our_mean = ipca.mean_.cpu().numpy()
        sklearn_mean = sklearn_pca.mean_
        
        assert np.allclose(our_mean, sklearn_mean, rtol=1e-10)


# =============================================================================
# Comparison with sklearn IncrementalPCA
# =============================================================================

class TestComparisonWithSklearnIPCA:
    """
    Compare against sklearn's IncrementalPCA.
    
    These should be closer than batch PCA comparison since they use
    similar incremental algorithms.
    """
    
    def test_components_match_sklearn_ipca(self, medium_data, device):
        """Components should match sklearn IncrementalPCA."""
        n_components = 20
        batch_size = 100
        
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_ipca.fit(medium_data)
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        our_components = ipca.components_.cpu().numpy()
        sklearn_components = sklearn_ipca.components_
        
        # Subspace should be very similar even if individual components differ
        angle = get_subspace_angle(our_components, sklearn_components)
        assert angle < 5.0, f"Subspace angle {angle}° too large"
    
    def test_explained_variance_ratio_close_to_sklearn_ipca(self, medium_data, device):
        """Explained variance ratio should be similar to sklearn."""
        n_components = 20
        batch_size = 100
        
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_ipca.fit(medium_data)
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        our_ratio = ipca.explained_variance_ratio_.cpu().numpy()
        sklearn_ratio = sklearn_ipca.explained_variance_ratio_
        
        # Total explained variance should be very close
        assert np.allclose(our_ratio.sum(), sklearn_ratio.sum(), rtol=0.05)
    
    def test_transform_similarity_with_sklearn_ipca(self, medium_data, device):
        """Transformed data should have similar properties to sklearn."""
        n_components = 20
        batch_size = 100
        
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_ipca.fit(medium_data)
        sklearn_transformed = sklearn_ipca.transform(medium_data)
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        our_transformed = ipca.transform(medium_data).numpy()
        
        # Variance of each component should be similar
        our_var = np.var(our_transformed, axis=0, ddof=1)
        sklearn_var = np.var(sklearn_transformed, axis=0, ddof=1)
        
        # Check that variances are similar (within 10%)
        assert np.allclose(our_var, sklearn_var, rtol=0.1)


# =============================================================================
# Batch Size Sensitivity Tests
# =============================================================================

class TestBatchSizeSensitivity:
    """
    Tests scrutinizing how batch_size affects accuracy.
    
    IMPORTANT: Incremental PCA inherently differs from batch PCA on random data
    due to the sequential nature of updates. The correct reference is sklearn's
    IncrementalPCA with the same batch size, which our implementation should match.
    """
    
    @pytest.mark.parametrize("batch_size", [10, 25, 50, 100, 200, 500])
    def test_components_match_sklearn_ipca_at_each_batch_size(
        self, medium_data, device, batch_size
    ):
        """
        Components should match sklearn IncrementalPCA for each batch size.
        
        This is the key accuracy test: our implementation should produce
        the same results as sklearn's IncrementalPCA with matching batch sizes.
        """
        n_components = 10
        
        # Reference: sklearn IncrementalPCA with same batch size
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_ipca.fit(medium_data)
        sklearn_components = sklearn_ipca.components_
        
        # Our implementation with same batch size
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        our_components = ipca.components_.cpu().numpy()
        
        # Should match sklearn IncrementalPCA very closely
        angle = get_subspace_angle(our_components, sklearn_components)
        assert angle < 1.0, f"batch_size={batch_size}: angle={angle}° vs sklearn IPCA"
    
    @pytest.mark.parametrize("batch_size", [10, 25, 50, 100, 200, 500])
    def test_explained_variance_ratio_matches_sklearn_ipca(
        self, medium_data, device, batch_size
    ):
        """
        Explained variance ratio should match sklearn IncrementalPCA.
        """
        n_components = 10
        
        # Reference: sklearn IncrementalPCA
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_ipca.fit(medium_data)
        sklearn_ratio = sklearn_ipca.explained_variance_ratio_
        
        # Our implementation
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        our_ratio = ipca.explained_variance_ratio_.cpu().numpy()
        
        # Total explained variance should be very close to sklearn IPCA
        assert np.allclose(our_ratio.sum(), sklearn_ratio.sum(), rtol=0.01)
    
    @pytest.mark.parametrize("batch_size", [10, 25, 50, 100, 200, 500])
    def test_reconstruction_error_matches_sklearn_ipca(
        self, medium_data, device, batch_size
    ):
        """
        Reconstruction error should match sklearn IncrementalPCA.
        """
        # Use n_components that works with all batch sizes (batch_size >= n_components)
        n_components = min(batch_size, 10)
        
        # Reference: sklearn IncrementalPCA
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_ipca.fit(medium_data)
        sklearn_rec = sklearn_ipca.inverse_transform(sklearn_ipca.transform(medium_data))
        sklearn_error = np.mean((medium_data - sklearn_rec) ** 2)
        
        # Our implementation
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        our_rec = ipca.inverse_transform(ipca.transform(medium_data)).numpy()
        our_error = np.mean((medium_data - our_rec) ** 2)
        
        # Reconstruction error should be very close
        rel_error = abs(our_error - sklearn_error) / sklearn_error if sklearn_error > 0 else 0
        assert rel_error < 0.01, f"batch_size={batch_size}: reconstruction rel_error={rel_error:.4f}"
    
    def test_very_small_batch_size_matches_sklearn(self, small_data, device):
        """
        Test with batch_size=5 (smallest that works with n_components=5).
        Even small batches should match sklearn IncrementalPCA.
        """
        # Note: sklearn requires batch_size >= n_components
        n_components = 5
        batch_size = 5
        
        # Reference: sklearn IncrementalPCA
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_ipca.fit(small_data)
        sklearn_components = sklearn_ipca.components_
        
        # Our implementation
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(small_data)
        our_components = ipca.components_.cpu().numpy()
        
        # Should still match sklearn closely
        angle = get_subspace_angle(our_components, sklearn_components)
        assert angle < 1.0, f"batch_size=5: angle={angle}° vs sklearn IPCA"
    
    def test_batch_size_larger_than_data(self, small_data, device):
        """Batch size larger than n_samples should work (single batch)."""
        n_components = 5
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=1000,  # Larger than small_data.shape[0]
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(small_data)
        
        # Should work and match sklearn PCA exactly
        sklearn_pca = PCA(n_components=n_components)
        sklearn_pca.fit(small_data)
        
        our_components = ipca.components_.cpu().numpy()
        sklearn_components = sklearn_pca.components_
        
        assert components_close(our_components, sklearn_components, rtol=1e-4)
    
    def test_partial_fit_accumulation(self, medium_data, device):
        """Multiple partial_fit calls should accumulate correctly."""
        n_components = 10
        batch_size = 100
        
        # Using fit method
        ipca_fit = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca_fit.fit(medium_data)
        
        # Manual partial_fit
        ipca_partial = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        for start in range(0, len(medium_data), batch_size):
            end = min(start + batch_size, len(medium_data))
            ipca_partial.partial_fit(medium_data[start:end])
        
        # Should produce identical results
        fit_components = ipca_fit.components_.cpu().numpy()
        partial_components = ipca_partial.components_.cpu().numpy()
        
        assert components_close(fit_components, partial_components, rtol=1e-10)
    
    @pytest.mark.parametrize("batch_size", [50, 100, 200])
    def test_transform_invariance_to_batch_size(self, medium_data, device, batch_size):
        """Transform result should not depend on batch_size used during transform."""
        n_components = 10
        
        # Fit with one batch size
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=100,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        # Transform with different batch sizes
        result_default = ipca.transform(medium_data)
        result_custom = ipca.transform(medium_data, batch_size=batch_size)
        
        assert torch.allclose(result_default, result_custom)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_n_components_equals_n_features(self, small_data, device):
        """Should work when keeping all components."""
        n_features = small_data.shape[1]
        
        ipca = IncrementalPCA(
            n_components=n_features,
            batch_size=50,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(small_data)
        
        assert ipca.n_components_ == n_features
        assert ipca.components_.shape[0] == n_features
    
    def test_n_components_none(self, small_data, device):
        """n_components=None should keep min(n_samples, n_features)."""
        n_samples, n_features = small_data.shape
        
        ipca = IncrementalPCA(
            n_components=None,
            batch_size=50,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(small_data)
        
        # First batch determines n_components_, limited by batch size
        expected = min(50, n_features)  # batch_size is 50
        assert ipca.n_components_ == expected
    
    def test_single_component(self, medium_data, device):
        """Should work with n_components=1."""
        ipca = IncrementalPCA(
            n_components=1,
            batch_size=100,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        sklearn_pca = PCA(n_components=1)
        sklearn_pca.fit(medium_data)
        
        our_var_ratio = ipca.explained_variance_ratio_.cpu().numpy()
        sklearn_var_ratio = sklearn_pca.explained_variance_ratio_
        
        # First component should explain similar variance
        # With incremental updates, there can be some deviation
        assert np.allclose(our_var_ratio, sklearn_var_ratio, rtol=0.25)
    
    def test_wide_data(self, wide_data, device):
        """Should work with n_features > n_samples."""
        n_samples, n_features = wide_data.shape
        n_components = min(n_samples - 1, 10)  # Limited by n_samples
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=10,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(wide_data)
        
        # Should fit without error
        assert ipca.components_.shape == (n_components, n_features)
    
    def test_tall_data(self, tall_data, device):
        """Should work well with n_samples >> n_features and match sklearn IPCA."""
        n_components = 10
        batch_size = 100
        
        # Reference: sklearn IncrementalPCA
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_ipca.fit(tall_data)
        sklearn_components = sklearn_ipca.components_
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(tall_data)
        our_components = ipca.components_.cpu().numpy()
        
        # Should match sklearn IPCA closely
        angle = get_subspace_angle(our_components, sklearn_components)
        assert angle < 1.0, f"Subspace angle {angle}° too large vs sklearn IPCA"


# =============================================================================
# Data Type Tests
# =============================================================================

class TestDataTypes:
    """Tests for different data types and input formats."""
    
    def test_float32(self, small_data, device):
        """Should work with float32 (default)."""
        data_32 = small_data.astype(np.float32)
        
        ipca = IncrementalPCA(
            n_components=10,
            batch_size=50,
            device=device,
            dtype=torch.float32,
        )
        ipca.fit(data_32)
        
        assert ipca.components_.dtype == torch.float32
    
    def test_float64(self, small_data, device):
        """Should work with float64."""
        data_64 = small_data.astype(np.float64)
        
        ipca = IncrementalPCA(
            n_components=10,
            batch_size=50,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(data_64)
        
        assert ipca.components_.dtype == torch.float64
    
    def test_numpy_input(self, small_data, device):
        """Should accept numpy arrays."""
        ipca = IncrementalPCA(n_components=10, batch_size=50, device=device)
        ipca.fit(small_data)  # small_data is numpy array
        
        assert ipca.components_ is not None
    
    def test_torch_input(self, small_data, device):
        """Should accept torch tensors."""
        torch_data = torch.tensor(small_data)
        
        ipca = IncrementalPCA(
            n_components=10, 
            batch_size=50, 
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(torch_data)
        
        assert ipca.components_ is not None
    
    def test_dtype_conversion(self, small_data, device):
        """Should convert input dtype to match model dtype."""
        data_32 = small_data.astype(np.float32)
        
        ipca = IncrementalPCA(
            n_components=10,
            batch_size=50,
            device=device,
            dtype=torch.float64,  # Different from input
        )
        ipca.fit(data_32)
        
        # Model should use float64
        assert ipca.components_.dtype == torch.float64


# =============================================================================
# Whitening Tests
# =============================================================================

class TestWhitening:
    """Tests for whitening functionality."""
    
    def test_whitened_variance(self, medium_data, device):
        """Whitened components should have unit variance."""
        ipca = IncrementalPCA(
            n_components=20,
            batch_size=100,
            whiten=True,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        X_whitened = ipca.transform(medium_data).numpy()
        
        # Each component should have variance close to 1
        variances = np.var(X_whitened, axis=0, ddof=1)
        assert np.allclose(variances, 1.0, rtol=0.1)
    
    def test_whitened_matches_sklearn(self, medium_data, device):
        """Whitened output should match sklearn PCA with whiten=True."""
        n_components = 20
        
        sklearn_pca = PCA(n_components=n_components, whiten=True)
        sklearn_pca.fit(medium_data)
        sklearn_whitened = sklearn_pca.transform(medium_data)
        
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=len(medium_data),
            whiten=True,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        our_whitened = ipca.transform(medium_data).numpy()
        
        # Align signs and compare
        for i in range(n_components):
            if np.dot(our_whitened[:, i], sklearn_whitened[:, i]) < 0:
                our_whitened[:, i] *= -1
        
        assert np.allclose(our_whitened, sklearn_whitened, rtol=1e-3)
    
    def test_whitening_inverse_transform(self, medium_data, device):
        """Inverse transform should undo whitening."""
        ipca = IncrementalPCA(
            n_components=20,
            batch_size=100,
            whiten=True,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(medium_data)
        
        X_whitened = ipca.transform(medium_data)
        X_reconstructed = ipca.inverse_transform(X_whitened)
        
        # Check that mean is approximately recovered
        reconstructed_mean = X_reconstructed.mean(dim=0).numpy()
        original_mean = medium_data.mean(axis=0)
        
        assert np.allclose(reconstructed_mean, original_mean, rtol=0.1)


# =============================================================================
# Correctness with Synthetic Data
# =============================================================================

class TestSyntheticData:
    """Tests with synthetic data having known properties."""
    
    def test_perfect_rank_k_data(self, device):
        """Test with data that is exactly rank k."""
        np.random.seed(42)
        n_samples, n_features, true_rank = 500, 50, 5
        
        # Create exactly rank-k data
        U = np.random.randn(n_samples, true_rank)
        V = np.random.randn(true_rank, n_features)
        X = U @ V
        
        ipca = IncrementalPCA(
            n_components=true_rank,
            batch_size=100,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(X)
        
        # First true_rank components should capture ~100% variance
        total_ratio = ipca.explained_variance_ratio_.sum().cpu().item()
        assert total_ratio > 0.999, f"Expected ratio > 0.999, got {total_ratio}"
    
    def test_reconstruction_of_low_rank_data(self, device):
        """Low-rank data should reconstruct nearly perfectly."""
        np.random.seed(42)
        n_samples, n_features, true_rank = 500, 50, 5
        
        U = np.random.randn(n_samples, true_rank)
        V = np.random.randn(true_rank, n_features)
        X = U @ V
        
        ipca = IncrementalPCA(
            n_components=true_rank,
            batch_size=100,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(X)
        
        X_reconstructed = ipca.inverse_transform(ipca.transform(X))
        mse = torch.mean((torch.tensor(X) - X_reconstructed) ** 2).item()
        
        assert mse < 1e-10, f"MSE {mse} too high for rank-k reconstruction"
    
    def test_with_correlated_data(self, correlated_data, device):
        """Test with data having structured correlations."""
        ipca = IncrementalPCA(
            n_components=10,
            batch_size=100,
            device=device,
            dtype=torch.float64,
        )
        ipca.fit(correlated_data)
        
        sklearn_pca = PCA(n_components=10)
        sklearn_pca.fit(correlated_data)
        
        # First few components should explain most variance
        our_ratio = ipca.explained_variance_ratio_.cpu().numpy()
        sklearn_ratio = sklearn_pca.explained_variance_ratio_
        
        # Sum of first 5 should explain >90% (data has 5 informative dims)
        assert our_ratio[:5].sum() > 0.9


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability in edge cases."""
    
    def test_zero_mean_feature(self, device):
        """Should handle features with zero mean."""
        np.random.seed(42)
        X = np.random.randn(200, 20)
        X -= X.mean(axis=0)  # Force zero mean
        
        ipca = IncrementalPCA(n_components=10, batch_size=50, device=device)
        ipca.fit(X)
        
        # Mean should be close to zero
        assert torch.allclose(
            ipca.mean_, 
            torch.zeros_like(ipca.mean_), 
            atol=1e-6
        )
    
    def test_constant_feature(self, device):
        """Should handle constant features (zero variance)."""
        np.random.seed(42)
        X = np.random.randn(200, 20)
        X[:, 0] = 5.0  # Constant feature
        
        ipca = IncrementalPCA(n_components=10, batch_size=50, device=device)
        ipca.fit(X)
        
        # Should fit without NaN
        assert not torch.any(torch.isnan(ipca.components_))
    
    def test_large_values(self, device):
        """Should handle large values."""
        np.random.seed(42)
        X = np.random.randn(200, 20) * 1e6
        
        ipca = IncrementalPCA(
            n_components=10, batch_size=50, device=device, dtype=torch.float64
        )
        ipca.fit(X)
        
        assert not torch.any(torch.isnan(ipca.components_))
        assert not torch.any(torch.isinf(ipca.components_))
    
    def test_small_values(self, device):
        """Should handle small values."""
        np.random.seed(42)
        X = np.random.randn(200, 20) * 1e-6
        
        ipca = IncrementalPCA(
            n_components=10, batch_size=50, device=device, dtype=torch.float64
        )
        ipca.fit(X)
        
        assert not torch.any(torch.isnan(ipca.components_))
