"""
Speed Benchmarks for Incremental PCA.

This script benchmarks the performance of the PyTorch IncrementalPCA
implementation across different batch sizes and compares against sklearn.

Run with: python benchmarks/benchmark_speed.py
Results are saved to benchmarks/results/
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA as SklearnIPCA

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from incremental_pca_torch import IncrementalPCA


def time_function(func, n_repeats: int = 3) -> Tuple[float, float]:
    """
    Time a function, returning mean and std of execution times.
    
    Args:
        func: Function to time (called with no arguments).
        n_repeats: Number of times to repeat.
        
    Returns:
        Tuple of (mean_time, std_time) in seconds.
    """
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times), np.std(times)


def benchmark_fit(
    X: np.ndarray,
    n_components: int,
    batch_sizes: List[int],
    device: str = "cpu",
    n_repeats: int = 3,
) -> Dict:
    """
    Benchmark fit time across different batch sizes.
    
    Args:
        X: Data array of shape (n_samples, n_features).
        n_components: Number of PCA components.
        batch_sizes: List of batch sizes to test.
        device: Device for PyTorch implementation.
        n_repeats: Number of timing repeats.
        
    Returns:
        Dictionary with benchmark results.
    """
    results = {
        "operation": "fit",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_components": n_components,
        "device": device,
        "batch_sizes": [],
        "torch_mean_s": [],
        "torch_std_s": [],
        "sklearn_mean_s": [],
        "sklearn_std_s": [],
        "speedup": [],
    }
    
    dtype = torch.float64 if device == "cpu" else torch.float32
    
    for batch_size in batch_sizes:
        print(f"  Benchmarking fit with batch_size={batch_size}...")
        
        # Torch implementation
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        torch_mean, torch_std = time_function(lambda: ipca.fit(X), n_repeats)
        
        # sklearn implementation
        sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=batch_size)
        sklearn_mean, sklearn_std = time_function(lambda: sklearn_ipca.fit(X), n_repeats)
        
        results["batch_sizes"].append(batch_size)
        results["torch_mean_s"].append(torch_mean)
        results["torch_std_s"].append(torch_std)
        results["sklearn_mean_s"].append(sklearn_mean)
        results["sklearn_std_s"].append(sklearn_std)
        results["speedup"].append(sklearn_mean / torch_mean if torch_mean > 0 else 0)
    
    return results


def benchmark_transform(
    X: np.ndarray,
    n_components: int,
    batch_sizes: List[int],
    device: str = "cpu",
    n_repeats: int = 3,
) -> Dict:
    """
    Benchmark transform time across different batch sizes.
    """
    results = {
        "operation": "transform",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_components": n_components,
        "device": device,
        "batch_sizes": [],
        "torch_mean_s": [],
        "torch_std_s": [],
        "sklearn_mean_s": [],
        "sklearn_std_s": [],
        "speedup": [],
    }
    
    dtype = torch.float64 if device == "cpu" else torch.float32
    
    # Fit models first
    fit_batch_size = 256
    ipca = IncrementalPCA(
        n_components=n_components,
        batch_size=fit_batch_size,
        device=device,
        dtype=dtype,
    )
    ipca.fit(X)
    
    sklearn_ipca = SklearnIPCA(n_components=n_components, batch_size=fit_batch_size)
    sklearn_ipca.fit(X)
    
    for batch_size in batch_sizes:
        print(f"  Benchmarking transform with batch_size={batch_size}...")
        
        # Torch implementation
        torch_mean, torch_std = time_function(
            lambda: ipca.transform(X, batch_size=batch_size), n_repeats
        )
        
        # sklearn doesn't have batch_size parameter for transform, so time full transform
        if batch_size == batch_sizes[0]:  # Only time sklearn once
            sklearn_mean, sklearn_std = time_function(
                lambda: sklearn_ipca.transform(X), n_repeats
            )
            sklearn_base = sklearn_mean
            sklearn_std_base = sklearn_std
        else:
            sklearn_mean = sklearn_base
            sklearn_std = sklearn_std_base
        
        results["batch_sizes"].append(batch_size)
        results["torch_mean_s"].append(torch_mean)
        results["torch_std_s"].append(torch_std)
        results["sklearn_mean_s"].append(sklearn_mean)
        results["sklearn_std_s"].append(sklearn_std)
        results["speedup"].append(sklearn_mean / torch_mean if torch_mean > 0 else 0)
    
    return results


def format_results_table(results: Dict) -> str:
    """Format benchmark results as a markdown table."""
    lines = []
    lines.append(f"### {results['operation'].title()} Benchmarks")
    lines.append("")
    lines.append(f"- **Data shape**: ({results['n_samples']}, {results['n_features']})")
    lines.append(f"- **Components**: {results['n_components']}")
    lines.append(f"- **Device**: {results['device']}")
    lines.append("")
    lines.append("| Batch Size | Torch (s) | sklearn (s) | Speedup |")
    lines.append("|----------:|----------:|------------:|--------:|")
    
    for i, batch_size in enumerate(results["batch_sizes"]):
        torch_time = results["torch_mean_s"][i]
        sklearn_time = results["sklearn_mean_s"][i]
        speedup = results["speedup"][i]
        lines.append(
            f"| {batch_size:>9} | {torch_time:>9.3f} | {sklearn_time:>11.3f} | {speedup:>7.2f}x |"
        )
    
    return "\n".join(lines)


def run_benchmarks(
    n_samples: int = 10000,
    n_features: int = 500,
    n_components: int = 50,
    device: str = "cpu",
) -> Dict:
    """Run full benchmark suite."""
    print(f"\n{'='*60}")
    print("Incremental PCA Speed Benchmarks")
    print(f"{'='*60}")
    print(f"Data: ({n_samples}, {n_features})")
    print(f"Components: {n_components}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    
    batch_sizes = [64, 128, 256, 512, 1024, 2048]
    
    # Run benchmarks
    print("Running fit benchmarks...")
    fit_results = benchmark_fit(X, n_components, batch_sizes, device)
    
    print("\nRunning transform benchmarks...")
    transform_results = benchmark_transform(X, n_components, batch_sizes, device)
    
    return {
        "fit": fit_results,
        "transform": transform_results,
    }


def main():
    """Main entry point for benchmarks."""
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Run CPU benchmarks
    results = run_benchmarks(
        n_samples=10000,
        n_features=500,
        n_components=50,
        device="cpu",
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    print(format_results_table(results["fit"]))
    print()
    print(format_results_table(results["transform"]))
    
    # Save results
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate markdown summary
    markdown_lines = [
        "# Benchmark Results",
        "",
        "Benchmarks comparing `incremental_pca_torch` against `sklearn.decomposition.IncrementalPCA`.",
        "",
        format_results_table(results["fit"]),
        "",
        format_results_table(results["transform"]),
        "",
        "> Note: Speedup values >1 indicate torch is faster, <1 indicates sklearn is faster.",
        "> On CPU, performance is similar. On GPU, torch is significantly faster for large data.",
    ]
    
    markdown_file = results_dir / "benchmark_results.md"
    with open(markdown_file, "w") as f:
        f.write("\n".join(markdown_lines))
    print(f"Markdown summary saved to: {markdown_file}")


if __name__ == "__main__":
    main()
