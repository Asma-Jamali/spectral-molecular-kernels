"""
Feature-removal ablation example on QM9.

For each representation, randomly removes N features and measures how Gaussian-kernel
KRR test MAE degrades. Covers two representations with different removal strategies
and N grids:

  ecfp6         features removed with probability proportional to their frequency in the training set
  selfies_ted  features removed uniformly at random
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Kermol.data.features import FingerprintGenerator, load_precomputed_rep
from Kermol.kernels.kernels import build_global_kernels, grid_search_cv_global, krr_predict
from Kermol.kernels.spectral import SpectralAnalyzer

lambda_grid = [10 ** (-3 * n) for n in range(1, 10)]
ablation_config = {
    'ecfp': {'n_values': [0, 64, 256], 'length_scales': [100., 1000., 10000.]},
    'selfies_ted': {'n_values': [0, 64, 256, 512], 'length_scales': [100., 1000., 10000.]},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rep', choices=['ecfp', 'selfies_ted'], required=True)
    p.add_argument('--property_name', default='gap', help='QM9 regression target.')
    p.add_argument('--n_train', type=int, default=1500)
    p.add_argument('--n_test', type=int, default=1500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cv_folds', type=int, default=4)
    p.add_argument('--data_path', required=True, help='Path to the dataset.')
    p.add_argument('--rep_dir', default=None, help="Directory containing selfies_ted.npy. Required when --rep is 'selfies_ted'.")
    args = p.parse_args()
    if args.rep == 'selfies_ted' and args.rep_dir is None:
        p.error("--rep_dir is required when --rep is 'selfies_ted'")
    return args


def split_indices(n_total, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    test_idx = rng.choice(indices, size=n_test, replace=False)
    remaining = np.setdiff1d(indices, test_idx)
    train_idx = rng.choice(remaining, size=n_train, replace=False)
    return train_idx, test_idx

def bit_prob(X):
    """Feature-frequency weights for probability-proportional removal (used for ecfp)."""
    p = X.sum(axis=0).astype(float)
    s = p.sum()
    return (p / s) if s > 0 else None


def remove_features(X, n, seed, prob=None):
    """Drop n feature columns. Uses prob weights when provided (ecfp), else uniform."""
    if n == 0:
        return X
    n = min(n, X.shape[1] - 1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[1], size=n, replace=False, p=prob)
    return np.delete(X, idx, axis=1)


def load_rep(rep_name, smiles, train_idx, test_idx, rep_dir=None):
    if rep_name == 'ecfp':
        gen = FingerprintGenerator(radius=3, n_bits=2048)
        X_full = np.array(gen.generate(smiles), dtype=np.float64)
    else:
        X_full = load_precomputed_rep('selfies_ted', rep_dir=rep_dir)
    return (np.asarray(X_full[train_idx], dtype=np.float64),
            np.asarray(X_full[test_idx],  dtype=np.float64))


def run_rep(rep_name, X_train_full, X_test_full, y_train, y_test, args):
    cfg = ablation_config[rep_name]
    prob = bit_prob(X_train_full) if rep_name == 'ecfp' else None
    rows = []
    for n in cfg['n_values']:
        X_train = remove_features(X_train_full, n, seed=args.seed + n, prob=prob)
        X_test = remove_features(X_test_full,  n, seed=args.seed + n, prob=prob)

        for ls in cfg['length_scales']:
            best = grid_search_cv_global(X_train, y_train, [ls], lambda_grid,'gaussian', cv=args.cv_folds, norm=2)
            K_train, K_test = build_global_kernels(X_train, X_test, ls, 'gaussian', norm=2)
            y_pred = krr_predict(K_train, y_train, K_test, best['lambda'])
            test_mae = float(np.mean(np.abs(y_pred - y_test)))
            metrics = SpectralAnalyzer(K_train).get_all_metrics(y=y_train, lam=best['lambda'])

            print(f"  {rep_name}  N={n:>4d}  ls={ls:<8.0f}  "
                  f"lambda={best['lambda']:.1e}  test_mae={test_mae:.4f}")
            rows.append({'rep': rep_name, 'N': n, 'length_scale': ls,
                         'lambda': best['lambda'], 'cv_mae': best['mae'],
                         'test_mae': test_mae, **metrics})
    return rows

output_dir = 'Results'


def main():
    args = parse_args()
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(args.data_path, allow_pickle=True)
    smiles = data['smiles']
    labels = data[args.property_name].astype(np.float64)

    train_idx, test_idx = split_indices(len(labels), args.n_train, args.n_test, args.seed)
    y_train, y_test = labels[train_idx], labels[test_idx]

    reps = [args.rep]

    all_rows = []
    for rep_name in reps:
        X_train_full, X_test_full = load_rep(rep_name, smiles, train_idx, test_idx, rep_dir=args.rep_dir)
        all_rows.extend(run_rep(rep_name, X_train_full, X_test_full, y_train, y_test, args))
    
    # Save results
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(output_dir, 'ablation_study.csv')
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for rep_name in reps:
        sub = df[df['rep'] == rep_name]
        best_per_n = sub.groupby('N')['test_mae'].min().reset_index().sort_values('N')
        ax.plot(best_per_n['N'], best_per_n['test_mae'], marker='o', label=rep_name)
    ax.set_xlabel('N removed features')
    ax.set_ylabel(f'Test MAE ({args.property_name})')
    ax.legend()
    out_png = os.path.join(output_dir, 'ablation_study.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
