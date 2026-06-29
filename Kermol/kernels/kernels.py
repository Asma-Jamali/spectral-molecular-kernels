import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from itertools import product

LOCAL_REPS        = {'soap', 'fchl19', 'acsf'}
FINGERPRINT_KERNELS = {
    'tanimoto', 'dice', 'otsuka', 'sogenfrei', 'braunblanquet',
    'faith', 'forbes', 'innerproduct', 'intersection', 'min_max', 'rand',
}


def _import_fp_kernels():
    import torch
    from gpytorch.kernels import ScaleKernel
    from gauche.kernels.fingerprint_kernels.tanimoto_kernel    import TanimotoKernel
    from gauche.kernels.fingerprint_kernels.dice_kernel         import DiceKernel
    from gauche.kernels.fingerprint_kernels.otsuka_kernel       import OtsukaKernel
    from gauche.kernels.fingerprint_kernels.sogenfrei_kernel    import SogenfreiKernel
    from gauche.kernels.fingerprint_kernels.braun_blanquet_kernel import BraunBlanquetKernel
    from gauche.kernels.fingerprint_kernels.faith_kernel        import FaithKernel
    from gauche.kernels.fingerprint_kernels.forbes_kernel       import ForbesKernel
    from gauche.kernels.fingerprint_kernels.inner_product_kernel import InnerProductKernel
    from gauche.kernels.fingerprint_kernels.intersection_kernel  import IntersectionKernel
    from gauche.kernels.fingerprint_kernels.minmax_kernel        import MinMaxKernel
    from gauche.kernels.fingerprint_kernels.rand_kernel          import RandKernel
    return torch, ScaleKernel, {
        'tanimoto':      TanimotoKernel,
        'dice':          DiceKernel,
        'otsuka':        OtsukaKernel,
        'sogenfrei':     SogenfreiKernel,
        'braunblanquet': BraunBlanquetKernel,
        'faith':         FaithKernel,
        'forbes':        ForbesKernel,
        'innerproduct':  InnerProductKernel,
        'intersection':  IntersectionKernel,
        'min_max':       MinMaxKernel,
        'rand':          RandKernel,
    }


def compute_fingerprint_kernel(X1: np.ndarray, X2: np.ndarray | None,
                                kernel_name: str) -> np.ndarray:
    """Compute a fingerprint kernel matrix using Gauche/PyTorch. No length scale."""
    torch, ScaleKernel, registry = _import_fp_kernels()
    if kernel_name not in registry:
        raise ValueError(f"Unknown fingerprint kernel: {kernel_name!r}. "
                         f"Choose from {sorted(registry)}")
    t1 = torch.tensor(X1, dtype=torch.float64)
    t2 = torch.tensor(X2, dtype=torch.float64) if X2 is not None else None
    km = ScaleKernel(registry[kernel_name]())
    with torch.no_grad():
        K = km(t1).evaluate() if t2 is None else km(t1, t2).evaluate()
    return K.detach().cpu().numpy()


def build_fingerprint_kernels(X_tr: np.ndarray, X_te: np.ndarray,
                               kernel_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (K_train, K_test) for fingerprint kernels. K_test: (n_test, n_train)."""
    K_tr = compute_fingerprint_kernel(X_tr, None,  kernel_name)
    K_te = compute_fingerprint_kernel(X_tr, X_te,  kernel_name).T
    return K_tr, K_te


def grid_search_cv_fingerprint(X: np.ndarray, y: np.ndarray,
                                lambda_grid: list, kernel_name: str,
                                cv: int = 4) -> dict:
    """
    CV grid search over lambda only (no length scale for fingerprint kernels).
    Precomputes the full n×n kernel once and slices it per fold.
    """
    kf     = KFold(n_splits=cv, shuffle=True, random_state=0)
    folds  = list(kf.split(X))

    K_full = compute_fingerprint_kernel(X, None, kernel_name)

    best_mae, best_params = np.inf, {}
    for lam in lambda_grid:
        fold_maes = []
        for tr, te in folds:
            K_tr   = K_full[np.ix_(tr, tr)]
            K_te   = K_full[np.ix_(tr, te)].T
            y_pred = krr_predict(K_tr, y[tr], K_te, lam)
            fold_maes.append(np.inf if y_pred is None
                             else np.mean(np.abs(y_pred - y[te])))
        mae = np.mean(fold_maes)
        if mae < best_mae:
            best_mae    = mae
            best_params = {'length': None, 'lambda': lam, 'mae': mae}

    return best_params


# ── Distance / kernel helpers (global) ───────────────────────────────────────

def cdist_metric(norm: int) -> str:
    if norm == 1:
        return 'cityblock'
    elif norm == 2:
        return 'euclidean'
    raise ValueError(f'Unsupported norm p={norm}; use 1 or 2')


def kernel_from_dist(dist, length, kernel):
    d = dist / length
    if kernel in ('gaussian', 'rbf'):
        return np.exp(-0.5 * d**2)
    elif kernel == 'laplacian':
        return np.exp(-d)
    raise ValueError(f'Unknown kernel: {kernel}')


def build_global_kernels(X_tr, X_te, length, kernel, norm=2):
    """Return (K_train, K_test) for global kernels. K_test shape: (n_test, n_train)."""
    if kernel == 'linear':
        K_tr = X_tr @ X_tr.T
        K_te = X_te @ X_tr.T
    else:
        metric = cdist_metric(norm)
        K_tr   = kernel_from_dist(cdist(X_tr, X_tr, metric=metric), length, kernel)
        K_te   = kernel_from_dist(cdist(X_tr, X_te, metric=metric), length, kernel).T
    return K_tr, K_te


def krr_predict(K_tr, y_tr, K_te, lam):
    """Solve (K + lam*I) alpha = y, return K_te @ alpha."""
    A = K_tr + np.eye(len(K_tr)) * lam
    try:
        L = scipy.linalg.cholesky(A, lower=True)
    except scipy.linalg.LinAlgError:
        return None
    alpha = scipy.linalg.cho_solve((L, True), y_tr)
    return K_te @ alpha


def krr_predict_zero_reg(K_tr: np.ndarray, y_tr: np.ndarray,
                          K_te: np.ndarray) -> np.ndarray:
    """Solve K alpha = y via least-squares (no regularization).

    Uses SVD-based lstsq so it handles rank-deficient kernel matrices
    (e.g. truncated eigenspaces with many zeroed eigenvalues).
    """
    alpha, _, _, _ = scipy.linalg.lstsq(K_tr, y_tr, cond=None)
    return K_te @ alpha


# ── Local kernel helpers (qml2) ───────────────────────────────────────────────

def _prepare_qml2_input(X):
    """Convert object array of per-molecule reps to (flat, natoms) required by qml2.
    X : (n_mols,) object array, each element shape (n_atoms_i, n_features).
    Returns:
        flat   : (total_atoms, n_features) float array
        natoms : (n_mols,) int32 array
    """
    flat   = np.concatenate(list(X))
    natoms = np.array([mol.shape[0] for mol in X], dtype=np.int32)
    return flat, natoms


def _prepare_qml2_charges(Z):
    """Concatenate per-molecule nuclear charge arrays for the dn kernel variants.
    Z : (n_mols,) object array, each element a 1-D int array of atomic numbers.
    Returns flat (total_atoms,) int32 array.
    """
    return np.concatenate(list(Z)).astype(np.int32)


def _import_qml2_kernels():
    try:
        from qml2.kernels import (
            local_gaussian_kernel,     local_gaussian_kernel_symmetric,
            local_laplacian_kernel,    local_laplacian_kernel_symmetric,
            local_matern_kernel,       local_matern_kernel_symmetric,
            local_dn_gaussian_kernel,  local_dn_gaussian_kernel_symmetric,
            local_dn_laplacian_kernel, local_dn_laplacian_kernel_symmetric,
            local_dn_matern_kernel,    local_dn_matern_kernel_symmetric,
        )
        return (local_gaussian_kernel,     local_gaussian_kernel_symmetric,
                local_laplacian_kernel,    local_laplacian_kernel_symmetric,
                local_matern_kernel,       local_matern_kernel_symmetric,
                local_dn_gaussian_kernel,  local_dn_gaussian_kernel_symmetric,
                local_dn_laplacian_kernel, local_dn_laplacian_kernel_symmetric,
                local_dn_matern_kernel,    local_dn_matern_kernel_symmetric)
    except ImportError:
        raise ImportError('qml2 is required for local kernel computations')


def _local_kernel_symmetric(flat, natoms, sigma, kernel, ncharges=None):
    """Build the full n×n symmetric local kernel matrix.
    If ncharges is provided, uses the dn (delta-nuclear-charge) variant so that
    only same-element atom pairs contribute — the physically correct choice.
    """
    (_, gk_sym, _, lk_sym, _, mk_sym,
     _, dn_gk_sym, _, dn_lk_sym, _, dn_mk_sym) = _import_qml2_kernels()

    if ncharges is not None:
        if kernel in ('gaussian', 'rbf'):
            return dn_gk_sym(flat, natoms, ncharges, sigma)
        elif kernel == 'laplacian':
            return dn_lk_sym(flat, natoms, ncharges, sigma)
        elif kernel in ('matern0', 'matern1', 'matern2'):
            return dn_mk_sym(flat, natoms, ncharges, sigma, order=int(kernel[-1]))
    else:
        if kernel in ('gaussian', 'rbf'):
            return gk_sym(flat, natoms, sigma)
        elif kernel == 'laplacian':
            return lk_sym(flat, natoms, sigma)
        elif kernel in ('matern0', 'matern1', 'matern2'):
            return mk_sym(flat, natoms, sigma, order=int(kernel[-1]))
    raise ValueError(f'Unknown local kernel: {kernel}. '
                     f'Choose from gaussian, laplacian, matern0, matern1, matern2.')


def build_local_kernels(X_tr, X_te, length, kernel,
                        Z_tr=None, Z_te=None):
    """Return (K_train, K_test) for final evaluation.
    K_test shape: (n_test, n_train).
    Z_tr / Z_te : optional object arrays of per-molecule nuclear charges.
    When provided, the dn (Kronecker-delta) kernel variant is used so that
    only same-element atom pairs are compared.
    """
    (gk, gk_sym, lk, lk_sym, mk, mk_sym,
     dn_gk, dn_gk_sym, dn_lk, dn_lk_sym,
     dn_mk, dn_mk_sym) = _import_qml2_kernels()

    A_flat, A_natoms = _prepare_qml2_input(X_tr)
    B_flat, B_natoms = _prepare_qml2_input(X_te)

    use_dn = Z_tr is not None and Z_te is not None
    if use_dn:
        A_ncharges = _prepare_qml2_charges(Z_tr)
        B_ncharges = _prepare_qml2_charges(Z_te)

    if kernel in ('gaussian', 'rbf'):
        if use_dn:
            K_tr = dn_gk_sym(A_flat, A_natoms, A_ncharges, length)
            K_te = dn_gk(A_flat, B_flat, A_natoms, B_natoms,
                         A_ncharges, B_ncharges, length).T
        else:
            K_tr = gk_sym(A_flat, A_natoms, length)
            K_te = gk(A_flat, B_flat, A_natoms, B_natoms, length).T
    elif kernel == 'laplacian':
        if use_dn:
            K_tr = dn_lk_sym(A_flat, A_natoms, A_ncharges, length)
            K_te = dn_lk(A_flat, B_flat, A_natoms, B_natoms,
                         A_ncharges, B_ncharges, length).T
        else:
            K_tr = lk_sym(A_flat, A_natoms, length)
            K_te = lk(A_flat, B_flat, A_natoms, B_natoms, length).T
    elif kernel in ('matern0', 'matern1', 'matern2'):
        order = int(kernel[-1])
        if use_dn:
            K_tr = dn_mk_sym(A_flat, A_natoms, A_ncharges, length, order=order)
            K_te = dn_mk(A_flat, B_flat, A_natoms, B_natoms,
                         A_ncharges, B_ncharges, length, order=order).T
        else:
            K_tr = mk_sym(A_flat, A_natoms, length, order=order)
            K_te = mk(A_flat, B_flat, A_natoms, B_natoms, length, order=order).T
    else:
        raise ValueError(f'Unknown local kernel: {kernel}. '
                         f'Choose from gaussian, laplacian, matern0, matern1, matern2.')

    return K_tr, K_te


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search_cv_global(X, y, length_grid, lambda_grid, kernel, cv=4, norm=2):
    kf      = KFold(n_splits=cv, shuffle=True, random_state=0)
    folds   = list(kf.split(X))
    y_folds = [(y[tr], y[te]) for tr, te in folds]

    best_mae, best_params = np.inf, {}

    if kernel == 'linear':
        # Dot products have no length hyperparameter — precompute once per fold
        kernels_cv = [(X[tr] @ X[tr].T, X[te] @ X[tr].T) for tr, te in folds]
        for lam in lambda_grid:
            fold_maes = []
            for k, (K_tr, K_te) in enumerate(kernels_cv):
                y_pred = krr_predict(K_tr, y_folds[k][0], K_te, lam)
                fold_maes.append(np.inf if y_pred is None
                                 else np.mean(np.abs(y_pred - y_folds[k][1])))
            mae = np.mean(fold_maes)
            if mae < best_mae:
                best_mae    = mae
                best_params = {'length': None, 'lambda': lam, 'mae': mae}
    else:
        metric = cdist_metric(norm)
        dists  = [
            (cdist(X[tr], X[tr], metric=metric),
             cdist(X[tr], X[te], metric=metric))
            for tr, te in folds
        ]
        for lam, length in product(lambda_grid, length_grid):
            fold_maes = []
            for k, (d_tr, d_te) in enumerate(dists):
                K_tr   = kernel_from_dist(d_tr, length, kernel)
                K_te   = kernel_from_dist(d_te, length, kernel).T
                y_pred = krr_predict(K_tr, y_folds[k][0], K_te, lam)
                fold_maes.append(np.inf if y_pred is None
                                 else np.mean(np.abs(y_pred - y_folds[k][1])))
            mae = np.mean(fold_maes)
            if mae < best_mae:
                best_mae    = mae
                best_params = {'length': length, 'lambda': lam, 'mae': mae}

    return best_params


def grid_search_cv_local(X, y, length_grid, lambda_grid, kernel, cv=4, Z=None):
    """CV grid search for local kernels using qml2.

    Efficient strategy (mirrors global kernel approach):
      - Precompute the full n×n kernel matrix ONCE per sigma value.
      - Slice it for each CV fold — reused across all lambda values.
      - Cost: |length_grid| kernel builds  (vs |length_grid|×|lambda_grid|×cv before).

    X : (n_mols,) object array of per-molecule rep matrices.
    y : (n_mols,) label array.
    Z : (n_mols,) object array of per-molecule nuclear charges (optional).
        When provided, uses the dn (Kronecker-delta) kernel variant.
    """
    kf      = KFold(n_splits=cv, shuffle=True, random_state=0)
    folds   = list(kf.split(X))
    y_folds = [(y[tr], y[te]) for tr, te in folds]

    flat, natoms = _prepare_qml2_input(X)
    ncharges     = _prepare_qml2_charges(Z) if Z is not None else None

    best_mae, best_params = np.inf, {}

    for length in length_grid:
        # Build full n×n kernel once — reused across all lambda values and all folds
        K_full = _local_kernel_symmetric(flat, natoms, length, kernel, ncharges)

        for lam in lambda_grid:
            fold_maes = []
            for tr, te in folds:
                K_tr   = K_full[np.ix_(tr, tr)]
                K_te   = K_full[np.ix_(tr, te)].T      # (n_te, n_tr)
                y_pred = krr_predict(K_tr, y[tr], K_te, lam)
                fold_maes.append(np.inf if y_pred is None
                                 else np.mean(np.abs(y_pred - y[te])))
            mae = np.mean(fold_maes)
            if mae < best_mae:
                best_mae    = mae
                best_params = {'length': length, 'lambda': lam, 'mae': mae}

    return best_params
