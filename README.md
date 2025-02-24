# PyTorch PCA

A PyTorch implementation of Principal Component Analysis (PCA) that exactly matches scikit-learn's implementation with default settings. This library provides GPU-accelerated PCA functionality with a scikit-learn compatible interface.

## Features

* **PCA**: Standard PCA implementation (`pca.py`)
* **Incremental PCA**: Memory-efficient version that processes data in batches (`incremental_pca.py`, _contributed by: [`yry`][yry]_)
* **GPU Acceleration**: Both implementations support CUDA for faster computation
* **scikit-learn Compatible API**: Uses familiar fit/transform methods

## Usage

```python
import torch
from pca import PCA

# Create data
X = torch.randn(100, 20)  # 100 samples, 20 features

# Initialize and fit PCA
pca = PCA(n_components=10)
pca.fit(X)

# Transform data
X_transformed = pca.transform(X)

# Or do both in one step
X_transformed = pca.fit_transform(X)

# Reconstruct original data
X_reconstructed = pca.inverse_transform(X_transformed)
```

For incremental PCA (processing data in batches):

```python
from incremental_pca import IncrementalPCA

# Initialize
ipca = IncrementalPCA(n_components=10, n_features=20)

# Process batches
for batch in data_batches:
    ipca.partial_fit(batch)

# Transform new data
transformed_data = ipca.transform(new_data)
```

## Installation

Copy the code from `pca.py` or `incremental_pca.py` into your project.

## Benchmarking

Run `benchmark.py` to compare performance between this implementation and scikit-learn's.

## References

* scikit-learn [PCA][]
* scikit-learn [IncrementalPCA][]

# Related Work

* `valentingol`'s [`torch_pca`](https://github.com/valentingol/torch_pca) appears to be
  a more full featured and faster (it chooses an appropriate PCA algorithm depending
  on input dimensions) alternative PCA implementation also matching scikit-learn.

[PCA]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
[IncrementalPCA]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
[yry]: https://github.com/YRYoung
