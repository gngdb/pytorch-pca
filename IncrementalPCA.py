import torch.nn as nn
import torch
import torch.nn.functional as F
from PCA import PCA


class IncrementalPCA(PCA):
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that leverages PyTorch for GPU acceleration.

    This class provides methods to fit the model on data incrementally in batches, and to transform new data
    based on the principal components learned during the fitting process.

    Attributes:
        n_components (int, optional): Number of components to keep.
        n_features (int, optional): Number of original components.
        mean: The mean of the data, if not given it will be calculated and updated from the batched data.
    """

    def __init__(self, n_components: int, n_features: int, mean=None):
        super(IncrementalPCA, self).__init__()
        assert n_components < n_features
        self.n_components = n_components
        self.n_features = n_features

        self.fixed_mean = mean is not None

        self.register_buffer('mean_', mean if self.fixed_mean else torch.zeros(n_features).float())
        self.register_buffer('components_', torch.zeros((n_components, n_features)).float())
        self.register_buffer('singular_values_', torch.zeros((n_components,)).float())

        self.register_buffer('N_', torch.tensor([0]))

    def _validate_data(self, X):
        assert X.shape[0] >= self.n_components
        assert X.shape[1] >= self.n_components
        assert X.device == self.N_.device

    def update_mean_var(self, X):
        """
        Computes the incremental mean and variance for the data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: new data mean, updated mean, updated total sample count.
        """
        # if X.shape[0] == 0:
        #     return last_mean, last_variance, last_sample_count
        # last_mean, last_var, last_N = self.mean_, self.var_, self.N_
        last_mean, last_N = self.mean_, self.N_

        new_N = X.shape[0]
        new_mean = torch.mean(X, dim=0)

        updated_N = last_N + new_N

        updated_mean = (last_N * last_mean + new_N * new_mean) / updated_N

        # delta_mean = new_mean - last_mean
        # new_sum_square = torch.sum((X - new_mean) ** 2, dim=0)
        # updated_var = (last_var * last_N + new_sum_square + delta_mean ** 2 * last_N * new_N / updated_N) / updated_N

        return new_mean, updated_mean, updated_N

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for deterministic output.

        This method ensures that the output remains consistent across different runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v

    def partial_fit(self, X, check=True):
        if check:
            self._validate_data(X)
        n_samples, n_features = X.shape

        if self.fixed_mean:
            X -= self.mean_
            n_total_samples = n_samples + self.N_

        else:
            new_mean, updated_mean, n_total_samples = self.update_mean_var(X)
            X -= new_mean

        if self.N_:
            if self.fixed_mean:
                mean_cor = torch.zeros_like(self.mean_)
            else:
                mean_cor = torch.sqrt((self.N_ / n_total_samples) * n_samples) * (self.mean_ - new_mean)
                self.mean_ = updated_mean

            X = torch.vstack((self.singular_values_.unsqueeze(1) * self.components_, X, mean_cor))

        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)

        self.N_ = n_total_samples
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]

        return self

    def transform(self, X):
        return torch.mm(X - self.mean_, self.components_.T)

    def forward(self, X, check=True):
        if self.training:
            return self.fit(X, check)
        else:
            if self.N_:
                return self.transform(X)
            raise RuntimeError('PCA has not been fitted')


if __name__ == '__main__':
    import numpy as np
    from sklearn.decomposition import IncrementalPCA as sklearn_IPCA
    from sklearn.datasets import make_classification

    n_components = 5
    batch_size = 20

    X, _ = make_classification(n_samples=100, n_features=20, random_state=0)
    X = torch.tensor(X, dtype=torch.float32)

    n_batches = X.shape[0] // batch_size

    sklearn_ipca = sklearn_IPCA(n_components=n_components)

    ipca = IncrementalPCA(n_components=n_components, n_features=X.shape[-1]).cuda().train()

    for i in range(n_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        X_batch = X[start_index:end_index]
        if i != n_batches - 1:
            sklearn_ipca.partial_fit(X_batch)
            ipca.partial_fit(X_batch.cuda())

    ipca.eval()
    print('testing saving and loading state dict')
    torch.save({'pca': ipca.state_dict()}, 'ipca.pkl')
    ipca = IncrementalPCA(n_components=n_components, n_features=X.shape[-1]).cuda()
    ipca.load_state_dict(torch.load('ipca.pkl')['pca'])

    X_reduced_sklearn = sklearn_ipca.transform(X_batch)
    X_reduced_custom = ipca.transform(X_batch.cuda())

    X_reduced_custom_np = X_reduced_custom.cpu().numpy()

    print("\nSklearn IncrementalPCA transformed data (first 5 samples):\n",
          X_reduced_sklearn[:5])
    print("Custom IncrementalPCA transformed data (first 5 samples):\n",
          X_reduced_custom_np[:5])
    equal = np.allclose(X_reduced_sklearn, X_reduced_custom_np)
    print('Sklearn and custom outputs are equal:', equal)

