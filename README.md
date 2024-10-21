If I try and write PCA from memory in PyTorch I always make a mistake so it
doesn't do exactly the same thing as scikit-learn's PCA with the same settings.
This is a minimal implementation of PCA that matches scikit-learn's with default
settings (run `pca.py` to test this).

Now includes:

* `pca.py`: matches sklearn's [PCA][]
* `incremental_pca.py`: matches sklearn's [IncrementalPCA][], _contributed by: [`yry`][yry]_

# Install

Open `pca.py`, copy the code you'd like to use and then paste it where you'd
like to use it.

# Related Work

* `valentingol`'s [`torch_pca`](https://github.com/valentingol/torch_pca) appears to be
  a more full featured and faster (it chooses an appropriate PCA algorithm depending
  on input dimensions) alternative PCA implementation also matching scikit-learn.

[PCA]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
[IncrementalPCA]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
[yry]: https://github.com/YRYoung
