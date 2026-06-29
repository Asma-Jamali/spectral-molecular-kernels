import numpy as np

from .kernels import krr_predict


# ── Log-spaced truncation scheme ──────────────────────────────────────────────

def build_schemes(n_train: int) -> dict[str, list[int]]:
    """35 log-spaced k-values from 1 to n_train — no gaps in the eigenspectrum."""
    k_all = np.unique(
        np.round(np.logspace(0, np.log10(n_train), 35)).astype(int)
    ).tolist()
    return {"LogSweep": k_all}


# ── Functional reconstruction helpers (used by train_truncate.py) ─────────────

def reconstruct_train(V: np.ndarray, lam: np.ndarray, k: int) -> np.ndarray:
    """K_train_r = (V_k * lam_k) @ V_k^T — broadcasting avoids forming diag."""
    Vk = V[:, :k]
    return (Vk * lam[:k]) @ Vk.T


def reconstruct_test(K_test: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    """K_test_r = (K_test @ V_k) @ V_k^T  [shape: n_test × n_train]."""
    Vk = V[:, :k]
    return (K_test @ Vk) @ Vk.T


def tune_lambda(
    V: np.ndarray,
    lam: np.ndarray,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    y_train: np.ndarray,
    lambda_grid: list[float],
    k: int,
) -> tuple[float, float]:
    """
    Grid-search λ on a held-out val split using eigenspace kernels.

    Truncated sub-kernels are formed directly from eigenvectors:
        K_tr_r  = (V_k[tr]  * lk) @ V_k[tr].T
        K_val_r = (V_k[val] * lk) @ V_k[tr].T
    Cost is O(n_tr² k + n_val n_tr k) — zero feature-space kernel work.

    Returns (best_lambda, best_val_mae).
    """
    Vk = V[:, :k]
    lk = lam[:k]

    Vk_tr  = Vk[tr_idx]    # (n_tr,  k)
    Vk_val = Vk[val_idx]   # (n_val, k)

    K_tr  = (Vk_tr  * lk) @ Vk_tr.T   # (n_tr,  n_tr)
    K_val = (Vk_val * lk) @ Vk_tr.T   # (n_val, n_tr)

    y_tr  = y_train[tr_idx]
    y_val = y_train[val_idx]

    best_lam, best_mae = lambda_grid[0], float('inf')
    for alpha in lambda_grid:
        y_pred = krr_predict(K_tr, y_tr, K_val, alpha)
        if y_pred is None:
            continue
        mae = float(np.mean(np.abs(y_pred - y_val)))
        if mae < best_mae:
            best_mae = mae
            best_lam = alpha

    return best_lam, best_mae
