"""
Pytest fixtures for Incremental PCA tests.
"""

from typing import Tuple

import numpy as np
import pytest
import torch


@pytest.fixture
def random_seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def set_seeds(random_seed: int):
    """Set random seeds for numpy and torch."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


@pytest.fixture
def small_data(set_seeds) -> np.ndarray:
    """
    Small dataset for quick tests.
    Shape: (200, 50) - 200 samples, 50 features.
    """
    return np.random.randn(200, 50).astype(np.float64)


@pytest.fixture
def medium_data(set_seeds) -> np.ndarray:
    """
    Medium-sized dataset for thorough tests.
    Shape: (1000, 100) - 1000 samples, 100 features.
    """
    return np.random.randn(1000, 100).astype(np.float64)


@pytest.fixture
def tall_data(set_seeds) -> np.ndarray:
    """
    Tall dataset (n_samples >> n_features).
    Shape: (2000, 20) - 2000 samples, 20 features.
    """
    return np.random.randn(2000, 20).astype(np.float64)


@pytest.fixture
def wide_data(set_seeds) -> np.ndarray:
    """
    Wide dataset (n_features > n_samples).
    Shape: (50, 200) - 50 samples, 200 features.
    """
    return np.random.randn(50, 200).astype(np.float64)


@pytest.fixture
def correlated_data(set_seeds) -> np.ndarray:
    """
    Data with known correlation structure.
    First few components should explain most variance.
    Shape: (500, 50).
    """
    n_samples, n_features = 500, 50
    n_informative = 5
    
    # Create low-rank structure
    latent = np.random.randn(n_samples, n_informative)
    loadings = np.random.randn(n_informative, n_features)
    signal = latent @ loadings
    
    # Add small noise
    noise = 0.1 * np.random.randn(n_samples, n_features)
    
    return (signal + noise).astype(np.float64)


@pytest.fixture
def device() -> str:
    """Device to use for tests (CPU for CI compatibility)."""
    return "cpu"
