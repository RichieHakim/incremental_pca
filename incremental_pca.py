"""
Incremental Principal Component Analysis (IPCA) in PyTorch.

This module provides a GPU-accelerated implementation of Incremental PCA that
mirrors the scikit-learn API. The implementation processes data in batches,
enabling analysis of datasets that do not fit in memory or GPU VRAM.

Algorithm Overview
------------------
Incremental PCA works by updating the principal components as new batches of
data arrive, rather than requiring all data to be present at once. The core
algorithm is based on Ross et al. (2008) and uses the following approach:

1. **Online Mean/Variance**: Welford's algorithm (also known as Chan's parallel
   algorithm) is used to compute running mean and variance in a numerically
   stable manner. This allows accurate estimation even when data arrives in
   small batches.

2. **Incremental SVD**: When a new batch arrives, the previous principal
   components are combined with the new centered data into an augmented matrix.
   A mean correction term is included to account for the shift in global mean.
   SVD of this augmented matrix yields the updated components.

3. **Deterministic Sign Flipping**: To ensure reproducibility across runs,
   singular vectors are flipped to have their largest absolute value positive.

Key differences from sklearn:
- Uses PyTorch tensors (GPU-compatible)
- Processes batches on the specified device (CPU or GPU)
- Returns transformed data on CPU by default to avoid GPU memory issues

References
----------
- Ross, D. A., Lim, J., Lin, R. S., & Yang, M. H. (2008). Incremental learning
  for robust visual tracking. International Journal of Computer Vision, 77(1),
  125-141.
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with
  randomness: Probabilistic algorithms for constructing approximate matrix
  decompositions. SIAM review, 53(2), 217-288.
- scikit-learn IncrementalPCA implementation:
  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html

Example
-------
::

    import torch
    import numpy as np
    from incremental_pca import IncrementalPCA

    # Create some data
    X = np.random.randn(10000, 100).astype(np.float32)

    # Fit incrementally on GPU
    ipca = IncrementalPCA(n_components=10, batch_size=256, device='cuda')
    ipca.fit(X)

    # Transform new data (returns CPU tensor)
    X_transformed = ipca.transform(X)

RH 2024
"""

from typing import Optional, Union

import math

import numpy as np
import torch
from tqdm.auto import tqdm


class IncrementalPCA:
    """
    Incremental Principal Component Analysis (IPCA) using PyTorch.

    This class provides a memory-efficient, GPU-compatible implementation of
    PCA that processes data in batches. It mirrors the scikit-learn API for
    easy adoption while leveraging PyTorch for GPU acceleration.

    The model can be fit incrementally via ``partial_fit`` for streaming data,
    or all at once via ``fit`` which internally batches the data.

    Attributes:
        components_ (torch.Tensor):
            Principal axes in feature space, shape ``(n_components, n_features)``.
            Rows are sorted by decreasing explained variance.
        mean_ (torch.Tensor):
            Per-feature empirical mean, shape ``(n_features,)``.
        var_ (torch.Tensor):
            Per-feature empirical variance (biased), shape ``(n_features,)``.
        singular_values_ (torch.Tensor):
            Singular values corresponding to each component, shape ``(n_components,)``.
        explained_variance_ (torch.Tensor):
            Variance explained by each component (unbiased), shape ``(n_components,)``.
        explained_variance_ratio_ (torch.Tensor):
            Fraction of total variance explained by each component, shape ``(n_components,)``.
        noise_variance_ (float or torch.Tensor):
            Estimated noise variance (average variance not captured by components).
        n_samples_seen_ (int):
            Total number of samples processed across all ``partial_fit`` calls.
        n_components_ (int):
            Actual number of components (may be less than ``n_components`` if
            limited by data dimensions).
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        batch_size: int = 128,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        whiten_eps: float = 1e-7,
        verbose: bool = False,
    ):
        """
        Initialize the Incremental PCA model.

        Args:
            n_components (Optional[int]):
                Number of principal components to keep. If ``None``, all
                components are kept (limited by ``min(n_samples, n_features)``
                seen during fitting).
            whiten (bool):
                If ``True``, the ``transform`` method will scale each component
                to have unit variance. This is useful when the transformed data
                will be used in models that assume isotropic noise.
            batch_size (int):
                Number of samples to process in each batch during ``fit`` and
                ``transform``. Larger values use more memory but may be faster.
                For ``partial_fit``, the entire input is processed as one batch.
            device (str):
                Device to perform computations on. Examples: ``'cpu'``,
                ``'cuda'``, ``'cuda:0'``, ``'mps'``.
            dtype (torch.dtype):
                Data type for internal computations. Use ``torch.float32`` for
                speed or ``torch.float64`` for precision.
            whiten_eps (float):
                Small constant added to explained variance during whitening to
                prevent division by zero when a component has near-zero variance.
            verbose (bool):
                If ``True``, display progress bars during ``fit`` and ``transform``.
        """
        self.n_components = n_components
        self.whiten = whiten
        self.batch_size = batch_size
        self.dtype = dtype
        self.whiten_eps = whiten_eps
        self.verbose = verbose

        self.device = torch.device(device)

        ## Model state attributes (populated during fitting)
        self.components_: Optional[torch.Tensor] = None
        self.mean_: Optional[torch.Tensor] = None
        self.var_: Optional[torch.Tensor] = None
        self.singular_values_: Optional[torch.Tensor] = None
        self.explained_variance_: Optional[torch.Tensor] = None
        self.explained_variance_ratio_: Optional[torch.Tensor] = None
        self.noise_variance_: Optional[Union[float, torch.Tensor]] = None
        self.n_samples_seen_: int = 0
        self.n_components_: Optional[int] = None

    def _validate_input(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Validate and convert input to a PyTorch tensor of the correct dtype.

        This method handles numpy arrays (including memory-mapped arrays)
        efficiently by using ``torch.as_tensor``, which avoids copying when
        possible.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Input data array. Can be a numpy array, numpy memmap, or
                PyTorch tensor.

        Returns:
            (torch.Tensor):
                X_validated (torch.Tensor):
                    Input converted to a PyTorch tensor with ``dtype=self.dtype``.
                    Note: The tensor is NOT moved to ``self.device`` here; that
                    is done just before computation to minimize memory transfers.
        """
        if not isinstance(X, torch.Tensor):
            ## torch.as_tensor avoids copying if X is already a compatible array
            ## This is especially important for numpy memmaps
            X = torch.as_tensor(X, dtype=self.dtype)
        if X.dtype != self.dtype:
            X = X.to(self.dtype)
        return X

    def fit(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: None = None,
    ) -> "IncrementalPCA":
        """
        Fit the model to data X using minibatch processing.

        This method resets any previous fitting state and processes the entire
        dataset in batches of size ``self.batch_size``. Each batch is moved to
        the compute device, processed via ``partial_fit``, and then discarded.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Training data of shape ``(n_samples, n_features)``. Can be a
                numpy array (including memory-mapped arrays) or PyTorch tensor.
            y (None):
                Ignored. Present for API compatibility with sklearn.

        Returns:
            (IncrementalPCA):
                self (IncrementalPCA):
                    The fitted model instance.

        Example:
            .. code-block:: python

                ipca = IncrementalPCA(n_components=10, batch_size=256)
                ipca.fit(X_train)
        """
        ## Reset state for fresh fitting
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.var_ = None
        self.components_ = None
        self.noise_variance_ = None

        ## Get shape without loading entire array into memory
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

        ## Iterate over data in chunks
        for start in tqdm(
            range(0, n_samples, self.batch_size),
            disable=not self.verbose,
            desc="Fitting IPCA",
        ):
            end = min(start + self.batch_size, n_samples)
            ## Slice from source array (could be on CPU or disk via memmap)
            X_batch_raw = X[start:end]
            self.partial_fit(X_batch_raw)

        return self

    def partial_fit(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: None = None,
    ) -> "IncrementalPCA":
        """
        Incrementally update the model with a batch of samples.

        This method updates the mean, variance, and principal components using
        the new data. It can be called repeatedly with new batches for streaming
        or out-of-core learning.

        The algorithm proceeds as follows:

        1. **Update running statistics** using Welford's/Chan's parallel
           algorithm for numerically stable online mean and variance.

        2. **Construct augmented matrix** combining:
           - Previous components scaled by their singular values
           - New batch centered by its batch mean
           - Mean correction term accounting for global mean shift

        3. **Compute SVD** of the augmented matrix to get new components.

        4. **Apply deterministic sign flip** for reproducibility: flip each
           singular vector so its largest absolute element is positive.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Batch of samples, shape ``(n_samples_batch, n_features)``.
            y (None):
                Ignored. Present for API compatibility with sklearn.

        Returns:
            (IncrementalPCA):
                self (IncrementalPCA):
                    The updated model instance.

        Note:
            Unlike ``fit``, this method does NOT reset the model state. Call
            ``fit`` first if you want to start fresh.

        References:
            - Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). Algorithms for
              computing the sample variance: Analysis and recommendations.
              The American Statistician, 37(3), 242-247.
        """
        ## Move batch to compute device
        X = self._validate_input(X).to(self.device)
        n_samples, n_features = X.shape

        ## Initialize on first pass
        if self.n_samples_seen_ == 0:
            ## Cap n_components at min(n_samples, n_features) if not specified
            self.n_components_ = (
                min(n_samples, n_features)
                if self.n_components is None
                else self.n_components
            )
            self.mean_ = torch.zeros(n_features, device=self.device, dtype=self.dtype)
            self.var_ = torch.zeros(n_features, device=self.device, dtype=self.dtype)

        ## =========================================================================
        ## Step 1: Update Mean and Variance (Welford's/Chan's parallel algorithm)
        ## =========================================================================
        ## This algorithm computes running statistics in a numerically stable way
        ## by tracking: (1) current mean, (2) sum of squared deviations.
        ## Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        col_mean = torch.mean(X, dim=0)  # Shape: (n_features,)
        col_var = torch.var(X, dim=0, unbiased=False)  # Biased variance for this batch
        n_total_samples = self.n_samples_seen_ + n_samples

        if self.n_samples_seen_ == 0:
            ## First batch: just use batch statistics directly
            self.mean_ = col_mean
            self.var_ = col_var
        else:
            ## Parallel/incremental variance update (Chan's method)
            col_batch_mean = col_mean
            prev_mean = self.mean_

            ## Update mean: weighted combination of previous and batch means
            self.mean_ = (
                self.n_samples_seen_ * prev_mean + n_samples * col_batch_mean
            ) / n_total_samples

            ## Update variance using parallel variance formula:
            ## Var(A ∪ B) = (n_A * Var_A + n_B * Var_B + correction) / n_total
            ## where correction = (n_A * n_B / n_total) * (mean_A - mean_B)^2
            prev_sum_sq = self.var_ * self.n_samples_seen_  # Sum of squared devs (biased)
            curr_sum_sq = col_var * n_samples
            delta_mean = prev_mean - col_batch_mean
            m_correction = (self.n_samples_seen_ * n_samples) / n_total_samples

            total_sum_sq = prev_sum_sq + curr_sum_sq + m_correction * (delta_mean**2)
            self.var_ = total_sum_sq / n_total_samples

        ## =========================================================================
        ## Step 2: Incremental SVD Update
        ## =========================================================================
        ## We construct an augmented matrix K and compute its SVD.
        ## For the first batch, K is just the centered data.
        ## For subsequent batches, K stacks:
        ##   - Previous components scaled by singular values
        ##   - New data centered by batch mean
        ##   - Mean correction vector to account for shift in global mean
        if self.n_samples_seen_ == 0:
            ## First batch: standard SVD of centered data
            X_centered = X - self.mean_  # Shape: (n_samples, n_features)
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        else:
            ## Subsequent batches: incremental SVD update
            prev_components = self.components_  # Shape: (n_components_, n_features)
            prev_singular = self.singular_values_  # Shape: (n_components_,)

            ## Center by batch mean (NOT global mean - the mean correction term handles this)
            X_centered = X - col_mean  # Shape: (n_samples, n_features)

            ## Mean correction vector:
            ## This accounts for the shift in global mean when combining old and new data.
            ## The factor sqrt(n_old * n_new / n_total) * (old_mean - new_mean) ensures
            ## the variance contribution from the mean shift is properly incorporated.
            mean_diff = torch.sqrt(
                torch.as_tensor(
                    (self.n_samples_seen_ * n_samples) / n_total_samples,
                    device=self.device,
                    dtype=self.dtype,
                )
            ) * (prev_mean - col_mean)
            mean_correction = mean_diff.unsqueeze(0)  # Shape: (1, n_features)

            ## Stack into augmented matrix K:
            ## Row 0 to n_components-1: Previous subspace (S * V^T from old SVD)
            ## Row n_components to n_components+n_samples-1: New centered data
            ## Last row: Mean correction
            ## Total rows: n_components_ + n_samples + 1
            K = torch.vstack(
                [
                    prev_singular.view(-1, 1) * prev_components,  # (n_components_, n_features)
                    X_centered,  # (n_samples, n_features)
                    mean_correction,  # (1, n_features)
                ]
            )

            ## SVD of augmented matrix yields updated components
            U, S, Vt = torch.linalg.svd(K, full_matrices=False)

        ## =========================================================================
        ## Step 3: Deterministic Sign Flip
        ## =========================================================================
        ## SVD is unique only up to sign flips (U[:, i] and Vt[i, :] can both be
        ## negated and still be valid). For reproducibility, we enforce that each
        ## row of Vt has its largest absolute value be positive.
        max_abs_cols = torch.argmax(torch.abs(Vt), dim=1)  # Index of max abs val per row
        signs = torch.sign(Vt[torch.arange(Vt.shape[0], device=self.device), max_abs_cols])
        ## Handle zero signs (shouldn't happen but be safe)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Vt = Vt * signs.view(-1, 1)

        ## =========================================================================
        ## Step 4: Update Model State
        ## =========================================================================
        k = self.n_components_
        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[:k]  # Shape: (n_components_, n_features)
        self.singular_values_ = S[:k]  # Shape: (n_components_,)

        ## Explained variance (unbiased estimate):
        ## Var = S^2 / (n - 1), where S are singular values
        self.explained_variance_ = (self.singular_values_**2) / (self.n_samples_seen_ - 1)

        ## Total variance (convert biased tracking to unbiased):
        ## We track biased variance, so multiply by n/(n-1) for unbiased estimate
        total_var = (
            torch.sum(self.var_) * self.n_samples_seen_ / (self.n_samples_seen_ - 1)
        )

        ## Explained variance ratio
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        ## Noise variance: average variance not explained by kept components
        if self.n_components_ < n_features and self.n_samples_seen_ > 1:
            noise_var = total_var - self.explained_variance_.sum()
            self.noise_variance_ = noise_var / (n_features - self.n_components_)
        else:
            self.noise_variance_ = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply dimensionality reduction to X.

        Projects the input data onto the principal component space. Processing
        is done in batches to avoid out-of-memory errors with large datasets.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Data to transform, shape ``(n_samples, n_features)``. The number
                of features must match the training data.
            batch_size (Optional[int]):
                Number of samples to process per batch. If ``None``, uses
                ``self.batch_size``.

        Returns:
            (torch.Tensor):
                X_transformed (torch.Tensor):
                    Transformed data of shape ``(n_samples, n_components_)``,
                    placed on CPU to avoid GPU memory accumulation.

        Raises:
            RuntimeError:
                If the model has not been fitted yet.

        Example:
            .. code-block:: python

                ipca = IncrementalPCA(n_components=10).fit(X_train)
                X_proj = ipca.transform(X_test)
        """
        if self.components_ is None:
            raise RuntimeError("Model must be fitted before transforming data.")

        ## Get sample count without loading full array
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

        ## Use default batch size if not specified
        batch_size = batch_size if batch_size is not None else self.batch_size

        results = []

        ## Process in chunks to avoid OOM
        for start in tqdm(
            range(0, n_samples, batch_size),
            disable=not self.verbose,
            desc="Transforming",
        ):
            end = min(start + batch_size, n_samples)

            ## Step 1: Slice from source (CPU/disk) before moving to device
            X_batch_raw = X[start:end]
            X_batch = self._validate_input(X_batch_raw).to(self.device)

            ## Step 2: Center and project onto principal component space
            X_batch = X_batch - self.mean_  # Center using fitted mean
            X_transformed = X_batch @ self.components_.T  # Project: (batch, n_components_)

            ## Step 3: Optionally whiten (scale to unit variance)
            if self.whiten:
                ## Divide by sqrt(explained_variance) to get unit variance
                ## Add eps for numerical stability
                scale = torch.sqrt(self.explained_variance_) + self.whiten_eps
                X_transformed = X_transformed / scale

            ## Step 4: Move result to CPU immediately to free GPU memory
            results.append(X_transformed.cpu())

        return torch.cat(results, dim=0)

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: None = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Fit the model to X and apply dimensionality reduction.

        This is equivalent to calling ``fit(X)`` followed by ``transform(X)``,
        but may be slightly more convenient.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Training data of shape ``(n_samples, n_features)``.
            y (None):
                Ignored. Present for API compatibility with sklearn.
            batch_size (Optional[int]):
                Batch size for the transform step. If ``None``, uses
                ``self.batch_size``.

        Returns:
            (torch.Tensor):
                X_transformed (torch.Tensor):
                    Transformed training data of shape ``(n_samples, n_components_)``,
                    placed on CPU.

        Example:
            .. code-block:: python

                ipca = IncrementalPCA(n_components=10)
                X_proj = ipca.fit_transform(X_train)
        """
        self.fit(X)
        return self.transform(X, batch_size=batch_size)

    def inverse_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Transform data back to its original space.

        Reconstructs the original features from the reduced representation.
        Note that this is lossy if ``n_components < n_features``.

        Args:
            X (Union[np.ndarray, torch.Tensor]):
                Transformed data of shape ``(n_samples, n_components_)``.
            batch_size (Optional[int]):
                Number of samples to process per batch. If ``None``, uses
                ``self.batch_size``.

        Returns:
            (torch.Tensor):
                X_reconstructed (torch.Tensor):
                    Reconstructed data of shape ``(n_samples, n_features)``,
                    placed on CPU.

        Raises:
            RuntimeError:
                If the model has not been fitted yet.

        Example:
            .. code-block:: python

                X_proj = ipca.transform(X)
                X_reconstructed = ipca.inverse_transform(X_proj)
                reconstruction_error = torch.mean((X - X_reconstructed) ** 2)
        """
        if self.components_ is None:
            raise RuntimeError("Model must be fitted before inverse transforming data.")

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        batch_size = batch_size if batch_size is not None else self.batch_size

        results = []

        for start in tqdm(
            range(0, n_samples, batch_size),
            disable=not self.verbose,
            desc="Inverse Transforming",
        ):
            end = min(start + batch_size, n_samples)

            X_batch_raw = X[start:end]
            X_batch = self._validate_input(X_batch_raw).to(self.device)

            ## If whitening was applied, undo it first
            if self.whiten:
                scale = torch.sqrt(self.explained_variance_) + self.whiten_eps
                X_batch = X_batch * scale

            ## Project back to original space and add mean
            X_reconstructed = X_batch @ self.components_ + self.mean_

            results.append(X_reconstructed.cpu())

        return torch.cat(results, dim=0)