import torch
import torch.nn as nn
import torch.nn.functional as F


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_

if __name__ == "__main__":
    import numpy as np
    from sklearn.decomposition import PCA as sklearn_PCA
    from sklearn import datasets
    iris = torch.tensor(datasets.load_iris().data)
    _iris = iris.numpy()
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        iris = iris.to(device)
        for n_components in (2, 4, None):
            _pca = sklearn_PCA(n_components=n_components).fit(_iris)
            _components = torch.tensor(_pca.components_)
            pca = PCA(n_components=n_components).to(device).fit(iris)
            components = pca.components_
            assert torch.allclose(components, _components.to(device))
            _t = torch.tensor(_pca.transform(_iris))
            t = pca.transform(iris)
            assert torch.allclose(t, _t.to(device))
        __iris = pca.inverse_transform(t)
        assert torch.allclose(__iris, iris)
    print("passed!")
