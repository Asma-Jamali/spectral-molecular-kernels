"""
truncated_kernel.py — spectral truncation example on QM9.

Builds the same full kernel as full_kernel.py (--mode global: BoB + Gaussian,
--mode local: ACSF + Gaussian with Kronecker-delta "dn" screening), eigendecomposes
K_train, then sweeps a log-spaced set of truncation ranks k: for each k, keep only the
top-k eigenpairs of K_train (Kermol.kernels.truncate.reconstruct_train/reconstruct_test),
re-tune lambda on a held-out validation split, and re-fit KRR. This shows how test error
degrades as the kernel's effective rank is reduced.

Usage
-----
    python examples/truncated_kernel.py --mode global --data_path /path/to/qm9_data.npz
    python examples/truncated_kernel.py --mode local --n_train 500 --n_test 500 \
        --n_val 150 --data_path /path/to/qm9_data.npz

Output
------
    <workdir>/truncated_kernel_<mode>.csv — one row per truncation rank k (+ the
    untruncated baseline), with test MAE/R² and the validation MAE used to pick lambda.
    <workdir>/truncated_kernel_<mode>.png — test MAE vs k, baseline as a dashed line.
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from Kermol.kernels.kernels import (
    build_global_kernels, build_local_kernels,
    grid_search_cv_global, grid_search_cv_local,
    krr_predict,
)
from Kermol.kernels.truncate import build_schemes, reconstruct_train, reconstruct_test, tune_lambda

from full_kernel import (
    GLOBAL_LENGTH_GRID, LOCAL_LENGTH_GRID, LAMBDA_GRID,
    strip_acsf_padding, split_indices, generate_rep_split,
)

TRUNC_LAMBDA_GRID = [10 ** (-n) for n in range(3, 16)]   # 1e-3 → 1e-15, per-k tuning


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['global', 'local'], default='global',
                   help="'global' = BoB + Gaussian; 'local' = ACSF + Gaussian (dn).")
    p.add_argument('--property_name', default='gap', help='QM9 regression target.')
    p.add_argument('--n_train', type=int, default=1500)
    p.add_argument('--n_test', type=int, default=1500)
    p.add_argument('--n_val', type=int, default=300,
                   help='Held out from n_train for per-k lambda tuning.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cv_folds', type=int, default=4)
    p.add_argument('--data_path', required=True, help='Path to the QM9 .npz dataset.')
    p.add_argument('--workdir', default=os.path.join(os.path.dirname(__file__), 'output'))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    print(f"Loading QM9 from {args.data_path} …")
    data    = np.load(args.data_path, allow_pickle=True)
    coords  = data['coords']
    charges = data['Z']
    labels  = data[args.property_name].astype(np.float64)

    train_idx, test_idx = split_indices(len(labels), args.n_train, args.n_test, args.seed)
    y_train, y_test = labels[train_idx], labels[test_idx]

    rng      = np.random.default_rng(args.seed)
    val_mask = rng.choice(args.n_train, size=args.n_val, replace=False)
    tr_mask  = np.setdiff1d(np.arange(args.n_train), val_mask)

    is_local = args.mode == 'local'
    rep_name = 'acsf' if is_local else 'bob'
    length_grid = LOCAL_LENGTH_GRID if is_local else GLOBAL_LENGTH_GRID

    print(f"Generating '{rep_name}' representation for "
          f"{args.n_train + args.n_test} sampled molecules …")
    X_train, X_test = generate_rep_split(rep_name, coords, charges, train_idx, test_idx)

    Z_train = Z_test = None
    if is_local:
        X_train = strip_acsf_padding(X_train, charges[train_idx])
        X_test  = strip_acsf_padding(X_test,  charges[test_idx])
        Z_train, Z_test = charges[train_idx], charges[test_idx]

    print(f"[Step 1] Grid-searching (sigma, lambda) via {args.cv_folds}-fold CV …")
    if is_local:
        best = grid_search_cv_local(X_train, y_train, length_grid, LAMBDA_GRID,
                                    'gaussian', cv=args.cv_folds, Z=Z_train)
    else:
        best = grid_search_cv_global(X_train, y_train, length_grid, LAMBDA_GRID,
                                     'gaussian', cv=args.cv_folds, norm=2)
    print(f"  sigma={best['length']:.4g}  lambda={best['lambda']:.2e}  CV MAE={best['mae']:.4f}")

    print("[Step 2] Building full K_train / K_test …")
    if is_local:
        K_train, K_test = build_local_kernels(X_train, X_test, best['length'], 'gaussian',
                                              Z_tr=Z_train, Z_te=Z_test)
    else:
        K_train, K_test = build_global_kernels(X_train, X_test, best['length'], 'gaussian', norm=2)

    print("[Step 3] Eigendecomposing K_train …")
    eigvals, eigvecs = scipy.linalg.eigh(K_train)
    order   = np.argsort(-eigvals)
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    print(f"  lambda_max={eigvals[0]:.4e}  lambda_min={eigvals[-1]:.4e}")

    print("[Step 4] Evaluating the untruncated baseline …")
    y_pred_full = krr_predict(K_train, y_train, K_test, best['lambda'])
    mae_full = float(np.mean(np.abs(y_pred_full - y_test)))
    r2_full  = float(r2_score(y_test, y_pred_full))
    print(f"  Baseline test MAE={mae_full:.4f}  R²={r2_full:.4f}")

    rows = [{'k': args.n_train, 'pct': 1.0, 'val_mae': best['mae'],
             'test_mae': mae_full, 'test_r2': r2_full}]

    print("[Step 5] Truncation sweep …")
    k_values = build_schemes(args.n_train)['LogSweep']
    for k in k_values:
        if k >= args.n_train:
            continue
        lam, val_mae = tune_lambda(eigvecs, eigvals, tr_mask, val_mask, y_train,
                                   TRUNC_LAMBDA_GRID, k)
        K_tr_r   = reconstruct_train(eigvecs, eigvals, k)
        K_test_r = reconstruct_test(K_test, eigvecs, k)
        y_pred   = krr_predict(K_tr_r, y_train, K_test_r, lam)
        test_mae = float(np.mean(np.abs(y_pred - y_test)))
        test_r2  = float(r2_score(y_test, y_pred))
        print(f"  k={k:>5d} ({k / args.n_train:5.1%})  lambda={lam:.1e}  "
              f"val_mae={val_mae:.4f}  test_mae={test_mae:.4f}  R²={test_r2:.4f}")
        rows.append({'k': k, 'pct': round(k / args.n_train, 4), 'val_mae': val_mae,
                     'test_mae': test_mae, 'test_r2': test_r2})

    df = pd.DataFrame(rows).sort_values('k')
    out_csv = os.path.join(args.workdir, f'truncated_kernel_{args.mode}.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")

    fig, ax = plt.subplots(figsize=(6, 4))
    sweep = df[df['k'] < args.n_train]
    ax.plot(sweep['k'], sweep['test_mae'], marker='o', label='Truncated KRR')
    ax.axhline(mae_full, color='gray', linestyle='--', label='Untruncated baseline')
    ax.set_xscale('log')
    ax.set_xlabel('Truncation rank k')
    ax.set_ylabel(f'Test MAE ({args.property_name})')
    ax.set_title(f'Spectral truncation — {rep_name}+gaussian ({args.mode})')
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(args.workdir, f'truncated_kernel_{args.mode}.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_png}")


if __name__ == '__main__':
    main()
