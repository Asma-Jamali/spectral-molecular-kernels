import numpy as np
import torch
import gpytorch
from typing import Optional, Union, Literal
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin, RBF
from sklearn.metrics.pairwise import linear_kernel
from scipy.spatial.distance import cdist, pdist, squareform
from gpytorch.kernels import ScaleKernel

from qml2.kernels import (
        local_laplacian_kernel,
        local_gaussian_kernel,
        local_laplacian_kernel_symmetric,
        local_gaussian_kernel_symmetric
    )
# from qmllib.kernels import wasserstein_kernel, laplacian_kernel, gaussian_kernel, linear_kernel, matern_kernel

class LaplaceKernel(StationaryKernelMixin, Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = float(length_scale)

    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        ell = self.length_scale

        if Y is None:
            # Pairwise L1 distances
            dists = pdist(X, metric='minkowski', p=1.0)
            dists = squareform(dists)
            K = np.exp(-dists / ell)
            np.fill_diagonal(K, 1.0)
        else:
            Y = np.atleast_2d(Y)
            dists = cdist(X, Y, metric='minkowski', p=1.0)
            K = np.exp(-dists / ell)

        return K

    def diag(self, X):
        # Laplace kernel is 1 on the diagonal
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True


    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True


class KernelGenerator:
    """
    Factory class to generate kernel matrices.
    Automatically switches between 'global' and 'local' kernels.
    """
    def __init__(self, kernel_name: str = 'laplacian', sigma: float = 1.0):
        """
        Args:
            kernel: 'laplacian' (or 'lap') or 'gaussian' (or 'rbf', 'gauss').
            sigma: Kernel width (length_scale).
        """
        self.kernel_name = kernel_name.lower()
        self.sigma = sigma


    def generate(
        self, 
        X1: np.ndarray, 
        X2: Optional[np.ndarray] = None, 
    ) -> np.ndarray:
        
        return self._compute_global_kernel(X1, X2)

    def _compute_global_kernel(self, X1, X2):
        """Internal handler for global representations (CM, BOB, etc)."""
        if self.kernel_name == 'linear':
            return linear_kernel(X1, X2)
        if self.kernel_name.startswith('lap'):
            kernel = LaplaceKernel(length_scale=self.sigma)
            # if X2 is None:
            #     kernel = laplacian_kernel(X1, X1, sigma=self.sigma)
            # else:
            #     kernel = laplacian_kernel(X1, X2, sigma=self.sigma)
        elif self.kernel_name.startswith('gauss') or self.kernel_name == 'rbf':
            kernel = RBF(length_scale=self.sigma)
            # if X2 is None:
            #     kernel = gaussian_kernel(X1, X1, sigma=self.sigma)
            # else:
            #     kernel = gaussian_kernel(X1, X2, sigma=self.sigma)
        else:
            raise ValueError(f"Unsupported global kernel type: {self.kernel_name}")
        
        # return kernel(X1, X2)
        return kernel

   