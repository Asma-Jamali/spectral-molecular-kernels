import numpy as np
import scipy.linalg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Tuple, Dict, Optional

class SpectralAnalyzer:
    """
    Analyzes the spectral properties of a Kernel matrix.
    Computes eigendecomposition once and calculates various spectral metrics.
    """

    def __init__(self, K: np.ndarray):
        """
        Args:
            K: (N, N) Symmetric Kernel Matrix.
        """
        self.n_samples = K.shape[0]
        
        # 1. Decompose and Sort (Largest to Smallest)
        eigval, eigvec = scipy.linalg.eigh(K)
        
        # np.save("debug_eigval.npy", eigval)
        
        # Sort descending (eigh returns ascending)
        idx = np.argsort(-eigval)
        self.eigenvalues = eigval[idx]
        self.eigenvectors = eigvec[:, idx]

        # Clamp negative eigenvalues (numerical noise) to epsilon for log calc
        self.eigenvalues_clean = np.where(self.eigenvalues <= 0, 1e-12, self.eigenvalues)

    def spectral_shannon_entropy(self) -> float:
        """
        Computes the Shannon Entropy of the normalized kernel spectrum.
        Higher entropy -> Flatter spectrum (more complex/effective dims).
        """
        eigenval_norm = self.eigenvalues_clean / np.sum(self.eigenvalues_clean)
        eigenval_norm = eigenval_norm[eigenval_norm > 0]  # Safety check
        
        # H(p) = - sum( p * log(p) )
        entropy = -np.sum(eigenval_norm * np.log(eigenval_norm))
        
        return np.exp(entropy)

    def intrinsic_dimension(self) -> float:
        """
        Computes Intrinsic Dimension defined as: Sum(evals) / Max(eval).
        """
        return np.sum(self.eigenvalues) / self.eigenvalues[0]

    def stable_rank(self) -> float:
        """
        Computes Stable Rank: Sum(evals^2) / Max(eval)^2.
        Ratio of Frobenius norm squared to Spectral norm squared.
        """
        return np.sum(self.eigenvalues**2) / (self.eigenvalues[0]**2)

    def compute_power_law_alpha(self) -> Tuple[float, float]:
        """
        Fits a power law (mu_i ~ i^-alpha) to the tail of the spectrum.
        Returns:
            alpha: The decay rate.
            r2: Goodness of fit.
        """
        # Prepare x (indices) and y (log eigenvalues)
        i = np.arange(1, len(self.eigenvalues_clean) + 1)
        log_i = np.log(i).reshape(-1, 1)
        log_mu = np.log(self.eigenvalues_clean)

        def fit_log_log(trim_frac):
            trim_n = int(len(log_i) * trim_frac)
            # keep the top eigenvalues.
            x_trim = log_i[:-trim_n]
            y_trim = log_mu[:-trim_n]
            
            reg = LinearRegression(fit_intercept=True).fit(x_trim, y_trim)
            alpha_val = -reg.coef_[0]
            r2_val = r2_score(y_trim, reg.predict(x_trim))
            return alpha_val, r2_val

        # Trim 50%
        alpha, r2 = fit_log_log(0.5)

        # Retry with 80% trim if fit is poor
        if r2 < 0.7:
            alpha, r2 = fit_log_log(0.8)

        return alpha, r2

    def target_weighted_eff_dim(self, y: np.ndarray, lam: float) -> float:
        """
        Computes Target-Weighted Effective Dimension:
        d_eff,y(λ) = Σ [ μ_j / (μ_j + λ) * ( (u_j^T y)^2 / ||y||^2 ) ]
        """
        # Normalize target
        y_norm_sq = np.dot(y, y)
        if y_norm_sq == 0:
            return 0.0

        # Projection of y onto each eigenvector (u_j^T y)
        # eigenvectors shape: (N, N), y shape: (N,)
        proj = self.eigenvectors.T @ y 

        # Compute weighted contributions
        weights = (self.eigenvalues / (self.eigenvalues + lam)) * ((proj**2) / y_norm_sq)

        return np.sum(weights)

    def get_all_metrics(self, y: Optional[np.ndarray] = None, lam: Optional[float] = None) -> Dict[str, float]:
        """
        Returns a dictionary containing all computed metrics.
        
        Args:
            y: (Optional) Target vector. If provided, target-weighted effective dimension is computed.
            lam: (Optional) Regularization parameter λ. Required ONLY if y is provided.
        """
        alpha, alpha_r2 = self.compute_power_law_alpha()
        
        metrics = {
            "SSE": self.spectral_shannon_entropy(),
            "ID": self.intrinsic_dimension(),
            "SR": self.stable_rank(),
            "alpha": alpha,
            "alpha_r2": alpha_r2
        }
        
        # Only compute target-weighted effective dimension if y is provided
        if y is not None:
            # Enforce that lambda is provided if y is provided
            if lam is None:
                raise ValueError("Regularization parameter 'lam' must be provided when calculating target-weighted metrics (y is provided).")
            
            metrics["tw_eff"] = self.target_weighted_eff_dim(y, lam)
            
        return metrics