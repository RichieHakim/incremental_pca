# Incremental PCA for PyTorch

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

Benchmarks comparing against `sklearn.decomposition.IncrementalPCA` on CPU.

**Configuration**: 10,000 samples × 500 features → 50 components

### Fit Performance

| Batch Size | Torch (s) | sklearn (s) | Speedup |
|----------:|----------:|------------:|--------:|
|        64 |     0.708 |       0.663 |    0.94x |
|       128 |     0.581 |       0.579 |    1.00x |
|       256 |     0.670 |       0.612 |    0.91x |
|       512 |     0.699 |       0.633 |    0.91x |
|      1024 |     0.585 |       0.548 |    0.94x |
|      2048 |     0.535 |       0.480 |    0.90x |

### Transform Performance

| Batch Size | Torch (s) | sklearn (s) | Speedup |
|----------:|----------:|------------:|--------:|
|        64 |     0.008 |       0.028 |    3.64x |
|       512 |     0.011 |       0.028 |    2.47x |
|      1024 |     0.007 |       0.028 |    3.72x |
|      2048 |     0.013 |       0.028 |    2.08x |

> **Note**: On CPU, performance is comparable to sklearn. The main advantage of this package is GPU acceleration, which provides significant speedups for large datasets.

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
