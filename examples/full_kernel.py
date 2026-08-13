"""
Full kernel ridge regression (KRR) with either a global or local representation.

  global  e.g., BoB representation + Gaussian kernel
  local   e.g., ACSF representation + Gaussian kernel, with Kronecker-delta 
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from Kermol.data.features import PhysicalGenerator
from Kermol.kernels.kernels import build_global_kernels, build_local_kernels, krr_predict
from Kermol.kernels.spectral import SpectralAnalyzer

global_length_grid = [10 ** n for n in range(2, 9)]       
local_length_grid = [0.1 * (2 ** n) for n in range(15)]   
lambda_grid = [10 ** (-3 * n) for n in range(1, 10)]  


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['global', 'local'], default='global', help="'global' = BoB + Gaussian; 'local' = ACSF + Gaussian (dn).")
    p.add_argument('--property_name', default='gap', help='QM9 regression target.')
    p.add_argument('--n_train', type=int, default=1500)
    p.add_argument('--n_test', type=int, default=1500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sigma', type=float, default=None, help='Kernel length scale.')
    p.add_argument('--lam', type=float, default=1e-6, help='Regularisation parameter.')
    p.add_argument('--data_path', required=True, help='Path to the QM9 .npz dataset.')
    return p.parse_args()


def strip_acsf_padding(raw, charges):
    """Convert ACSF's dense (n_mols, pad, n_feat) output to an object array of 
    per-molecule (n_atoms_i, n_feat) matrices — the shape build_local_kernels expects."""
    X = np.empty(len(charges), dtype=object)
    for i, mol_z in enumerate(charges):
        X[i] = raw[i, :len(mol_z), :]
    return X


def split_indices(n_total, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    test_idx = rng.choice(indices, size=n_test, replace=False)
    remaining = np.setdiff1d(indices, test_idx)
    train_idx = rng.choice(remaining, size=n_train, replace=False)
    return train_idx, test_idx


def generate_rep_split(rep_name, coords, charges, train_idx, test_idx):
    """Generate a representation once over train+test combined, then split."""
    n_train = len(train_idx)
    combined_idx = np.concatenate([train_idx, test_idx])
    gen = PhysicalGenerator(representation=rep_name)
    X_full = gen.generate(coords[combined_idx], charges[combined_idx])
    return X_full[:n_train], X_full[n_train:]

output_dir = "Results"

def main():
    args = parse_args()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading QM9 from {args.data_path} …")
    data = np.load(args.data_path, allow_pickle=True)
    coords = data['coords']
    charges = data['Z']
    labels = data[args.property_name].astype(np.float64)

    train_idx, test_idx = split_indices(len(labels), args.n_train, args.n_test, args.seed)
    y_train, y_test = labels[train_idx], labels[test_idx]

    is_local = args.mode == 'local'
    rep_name = 'acsf' if is_local else 'bob'
    sigma = args.sigma if args.sigma is not None else (2.0 if is_local else 1000.0)

    print(f"Generating '{rep_name}' representation for "
          f"{args.n_train + args.n_test} sampled molecules …")
    X_train, X_test = generate_rep_split(rep_name, coords, charges, train_idx, test_idx)

    Z_train = Z_test = None
    if is_local:
        X_train = strip_acsf_padding(X_train, charges[train_idx])
        X_test = strip_acsf_padding(X_test,  charges[test_idx])
        Z_train, Z_test = charges[train_idx], charges[test_idx]

    if is_local:
        K_train, K_test = build_local_kernels(X_train, X_test, sigma, 'gaussian', Z_tr=Z_train, Z_te=Z_test)
    else:
        K_train, K_test = build_global_kernels(X_train, X_test, sigma, 'gaussian', norm=2)

    y_pred = krr_predict(K_train, y_train, K_test, args.lam)
    test_mae = float(np.mean(np.abs(y_pred - y_test)))
    test_r2 = float(r2_score(y_test, y_pred))

    metrics = SpectralAnalyzer(K_train, center=True).get_all_metrics()
    print(f" SSE={metrics['SSE']:.4f}  ID={metrics['ID']:.4f}  SR={metrics['SR']:.4f}  "
          f"alpha={metrics['alpha']:.4f}")

    out_csv = os.path.join(output_dir, f'full_kernel_{args.mode}.csv')
    pd.DataFrame([{
        'mode': args.mode, 'rep': rep_name, 'kernel': 'gaussian',
        'property': args.property_name, 'n_train': args.n_train, 'n_test': args.n_test,
        'sigma': sigma, 'lambda': args.lam,
        'test_mae': test_mae, 'test_r2': test_r2, **metrics,
    }]).to_csv(out_csv, index=False)


if __name__ == '__main__':
    main()
