# Incremental PCA for PyTorch (Huge PCA on GPUs!)

[![PyPI version](https://badge.fury.io/py/incremental-pca-torch.svg)](https://badge.fury.io/py/incremental-pca-torch)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Incremental Principal Component Analysis (PCA) using PyTorch: This package provides a scikit-learn compatible API for performing PCA. This allows for PCA to be performed on datasets that are too large to fit in memory.

## Features

- **GPU Acceleration**: Perform PCA on GPUs for significant speedups on large datasets
- **Memory Efficient**: Process data in batches to handle datasets larger than available RAM/VRAM ("out of core")
- **sklearn Compatible**: Drop-in replacement with familiar `fit`, `transform`, `fit_transform` API
- **Streaming Support**: Use `partial_fit` for online learning from data streams
- **Lazy arrays / `numpy` memmap Support**: Efficiently process arrays on disk and memory-mapped files

## Installation

```bash
pip install incremental-pca-torch
```

### From Source

```bash
git clone https://github.com/RichieHakim/incremental_pca.git
cd incremental_pca
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from incremental_pca_torch import IncrementalPCA

# Create some data
X = np.random.randn(10000, 500).astype(np.float32)

# Fit incrementally using GPU
ipca = IncrementalPCA(
    n_components=50, 
    batch_size=256, 
    device='cuda'  # Use 'cpu' if no GPU available
)
ipca.fit(X)

# Transform new data
X_transformed = ipca.transform(X)
print(f"Reduced shape: {X_transformed.shape}")  # (10000, 50)

# Reconstruct data
X_reconstructed = ipca.inverse_transform(X_transformed)
```

### Streaming Data with `partial_fit`

```python
# For streaming or very large datasets
ipca = IncrementalPCA(n_components=50, device='cuda')

# Process data in chunks
for chunk in data_generator():
    ipca.partial_fit(chunk)

# Use the fitted model
X_transformed = ipca.transform(new_data)
```

### Using with Memory-Mapped Arrays

```python
import numpy as np

# Memory-mapped files work seamlessly
X_mmap = np.load('large_data.npy', mmap_mode='r')

ipca = IncrementalPCA(n_components=50, batch_size=256, device='cuda')
ipca.fit(X_mmap)  # Loads only one batch at a time
```

## API Reference

### `IncrementalPCA`

```python
IncrementalPCA(
    n_components=None,     # Number of components (default: min(n_samples, n_features))
    whiten=False,          # Scale components to unit variance
    batch_size=128,        # Samples per batch for fit/transform
    device='cpu',          # 'cpu', 'cuda', 'cuda:0', 'mps', etc.
    dtype=torch.float32,   # torch.float32 or torch.float64
    whiten_eps=1e-7,       # Numerical stability for whitening
    verbose=False,         # Show progress bars
)
```

### Methods

| Method | Description |
|--------|-------------|
| `fit(X)` | Fit the model to data X in batches |
| `partial_fit(X)` | Incrementally update model with a single batch |
| `transform(X)` | Project data onto principal components |
| `inverse_transform(X)` | Reconstruct data from components |
| `fit_transform(X)` | Fit and transform in one call |

### Attributes (after fitting)

| Attribute | Description |
|-----------|-------------|
| `components_` | Principal axes, shape `(n_components, n_features)` |
| `mean_` | Per-feature mean, shape `(n_features,)` |
| `explained_variance_` | Variance per component |
| `explained_variance_ratio_` | Fraction of total variance per component |
| `n_samples_seen_` | Total samples processed |

## Benchmarks

Benchmarks comparing against `sklearn.decomposition.IncrementalPCA` and `sklearn.decomposition.PCA` (Vanilla) on CPU.

**Configuration**: 10,000 samples × 500 features → 50 components

### Fit Performance

- **Vanilla PCA (sklearn)**: 0.128 s

| Batch Size | Torch (s) | sklearn IPCA (s) | Speedup vs IPCA | Speedup vs PCA |
|----------:|----------:|-----------------:|---------------:|---------------:|
|        64 |     0.713 |            0.659 |          0.92x |          0.18x |
|       128 |     0.561 |            0.560 |          1.00x |          0.23x |
|       256 |     0.648 |            0.599 |          0.92x |          0.20x |
|       512 |     0.687 |            0.610 |          0.89x |          0.19x |
|      1024 |     0.568 |            0.527 |          0.93x |          0.23x |
|      2048 |     0.471 |            0.442 |          0.94x |          0.27x |

### Transform Performance

- **Vanilla PCA (sklearn)**: 0.023 s

| Batch Size | Torch (s) | sklearn IPCA (s) | Speedup vs IPCA | Speedup vs PCA |
|----------:|----------:|-----------------:|---------------:|---------------:|
|        64 |     0.011 |            0.019 |          1.76x |          2.10x |
|       128 |     0.068 |            0.019 |          0.28x |          0.34x |
|       256 |     0.024 |            0.019 |          0.80x |          0.95x |
|       512 |     0.007 |            0.019 |          2.60x |          3.10x |
|      1024 |     0.007 |            0.019 |          2.67x |          3.18x |
|      2048 |     0.018 |            0.019 |          1.06x |          1.26x |

> **Note**: On CPU, incremental fitting is generally slower than full-batch randomized PCA when data fits in memory, due to the overhead of repeated SVDs. However, `incremental_pca_torch` enables processing datasets that **do not fit in memory** (or GPU memory), which is the primary use case. Transform speed varies by batch size but can be significantly faster than sklearn.

## Algorithm

This implementation uses the incremental SVD algorithm from Ross et al. (2008), which:

1. **Updates running statistics** using Welford's algorithm for numerically stable online mean and variance computation
2. **Constructs an augmented matrix** combining previous components with new centered data
3. **Performs SVD** on the augmented matrix to update components
4. **Applies deterministic sign flipping** for reproducibility

The algorithm matches sklearn's `IncrementalPCA` implementation exactly (verified via comprehensive test suite).

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

The test suite includes:
- Comparison against sklearn `PCA` (full-batch mode)
- Comparison against sklearn `IncrementalPCA` (various batch sizes)
- Batch size sensitivity tests
- Whitening tests
- Numerical stability tests
- Edge case handling

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Ross, D. A., Lim, J., Lin, R. S., & Yang, M. H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77(1), 125-141.
- [scikit-learn IncrementalPCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)
