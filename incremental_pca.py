import torch.nn as nn
import torch
import torch.nn.functional as F
from pca import PCA


def incremental_mean(X, last_mean, last_N):
        """
        Computes the incremental mean for the data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples,
              n_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: new data mean, updated mean,
              updated total sample count.
        """
        new_N = X.shape[0]
        new_mean = torch.mean(X, dim=0)

        # https://github.com/scikit-learn/scikit-learn/blob/9e38cd00d032f777312e639477f1f52f3ea4b3b7/sklearn/utils/extmath.py#L1106
        updated_N = last_N + new_N

        # https://github.com/scikit-learn/scikit-learn/blob/9e38cd00d032f777312e639477f1f52f3ea4b3b7/sklearn/utils/extmath.py#L1108
        updated_mean = (last_N * last_mean + new_N * new_mean) / updated_N

        return new_mean, updated_mean, updated_N


class IncrementalPCA(PCA):
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that
    leverages PyTorch for GPU acceleration.

    This class provides methods to fit the model on data incrementally in
    batches, and to transform new data based on the principal components learned
    during the fitting process.

    Attributes:
        n_components (int, optional): Number of components to keep.
        n_features (int, optional): Number of original components.
        mean: The mean of the data, if not given it will be calculated and
          updated from the batched data.
    """

    def __init__(self, n_components: int, n_features: int, mean=None):
        super(IncrementalPCA, self).__init__(n_components=n_components)
        assert n_components < n_features
        self.n_components = n_components
        self.n_features = n_features

        self.fixed_mean = mean is not None

        self.register_buffer('mean_', mean if self.fixed_mean
                             else torch.zeros(n_features).float())
        self.register_buffer('components_',
            torch.zeros((n_components, n_features), dtype=torch.float32))
        self.register_buffer('singular_values_',
            torch.zeros((n_components,), dtype=torch.float32))
        self.register_buffer('N_', torch.tensor([0]))

    def _validate_data(self, X):
        assert X.shape[0] >= self.n_components
        assert X.shape[1] >= self.n_components
        assert X.device == self.N_.device

    def _svd(self, X):
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)
        return U, S, Vt

    def partial_fit(self, X, check=True):
        if check:
            self._validate_data(X)
        n_samples, n_features = X.shape

        if self.fixed_mean:
            X -= self.mean_
            n_total_samples = n_samples + self.N_
        else:
            new_mean, updated_mean, n_total_samples = incremental_mean(X, self.mean_, self.N_)
            X -= new_mean

        if self.N_:
            if self.fixed_mean:
                mean_cor = torch.zeros_like(self.mean_)
            else:
                mean_cor = torch.sqrt(
                    (self.N_ / n_total_samples) * n_samples
                ) * (self.mean_ - new_mean)

            X = torch.vstack(
                (self.singular_values_.unsqueeze(1) * self.components_,
                 X,
                 mean_cor)
            )

        U, S, Vt = self._svd(X)

        self.N_ = n_total_samples
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]
        self.mean_ = updated_mean

        return self

    def transform(self, X):
        print(X.dtype, self.mean_.dtype, self.components_.dtype)
        return torch.mm(X - self.mean_, self.components_.T)

    def forward(self, X, check=True):
        if self.training:
            return self.fit(X, check)
        else:
            if self.N_:
                return self.transform(X)
            raise RuntimeError('PCA has not been fitted')


class _TestableIncrementalPCA(IncrementalPCA):
    def _svd(self, X):
        from scipy import linalg
        U, S, Vt = linalg.svd(X.numpy(), full_matrices=False, check_finite=False)
        U, Vt = sklearn.utils.extmath.svd_flip(U, Vt, u_based_decision=False)
        return torch.tensor(U), torch.tensor(S), torch.tensor(Vt)


if __name__ == '__main__':
    import numpy as np
    from sklearn.decomposition import IncrementalPCA as sklearn_IPCA
    from sklearn.datasets import make_classification
    import sklearn.utils

    n_components = 5
    batch_size = 20
    dtype = torch.float64

    X, _ = make_classification(n_samples=100, n_features=20, random_state=0)
    X = torch.tensor(X, dtype=dtype)

    n_batches = X.shape[0] // batch_size

    sklearn_ipca = sklearn_IPCA(n_components=n_components, batch_size=20)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    ipca = _TestableIncrementalPCA(n_components=n_components,
                                   n_features=X.shape[-1]).to(device)
    ipca = ipca.to(dtype)
    ipca.train()

    for i in range(n_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        X_batch = X[start_index:end_index]
        if i != n_batches - 1:
            if hasattr(sklearn_ipca, 'n_samples_seen_'):
                m, v, n = sklearn_ipca.mean_, sklearn_ipca.var_, sklearn_ipca.n_samples_seen_
            else:
                m, v, n = [np.zeros((X.shape[-1],))] * 3
            um, _, uc = sklearn.utils.extmath._incremental_mean_and_var(X_batch.numpy(), m, v, n)
            _nm, _um, _n = incremental_mean(X_batch, torch.tensor(m), torch.tensor(n))
            assert np.allclose(um, _um.numpy()) # unit test mean func
            sklearn_ipca.partial_fit(X_batch)
            ipca.partial_fit(X_batch.to(device))

    ipca.eval()
    print('testing saving and loading state dict')
    torch.save({'pca': ipca.state_dict()}, 'ipca.pkl')
    ipca = _TestableIncrementalPCA(n_components=n_components,
                                   n_features=X.shape[-1]).to(device)
    ipca.load_state_dict(torch.load('ipca.pkl')['pca'])
    ipca.to(dtype)

    X_reduced_sklearn = sklearn_ipca.transform(X_batch)
    X_reduced_custom = ipca.transform(X_batch.to(device))

    X_reduced_custom_np = X_reduced_custom.cpu().numpy()

    print("\nSklearn IncrementalPCA transformed data (first 5 samples):\n",
          X_reduced_sklearn[:5])
    print("Custom IncrementalPCA transformed data (first 5 samples):\n",
          X_reduced_custom_np[:5])
    equal = np.allclose(X_reduced_sklearn, X_reduced_custom_np)
    err = np.abs(X_reduced_sklearn - X_reduced_custom_np).max()
    assert equal, f'Error: {err}'
    print('Sklearn and custom outputs are equal:', equal)

