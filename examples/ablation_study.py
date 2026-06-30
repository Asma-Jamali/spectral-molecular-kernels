"""
ablation_study.py — feature-removal ablation example on QM9.

For each representation, randomly removes N features and measures how Gaussian-kernel
KRR test MAE and kernel spectral metrics degrade. Covers two representations whose
useful N (features-removed) ranges differ a lot, so each gets its own grid:

  bob          Bag-of-Bonds vector (a few hundred dims, grows with sample diversity)
                                                  -> N in {0, 16, 64}
  selfies_ted  1024-dim pretrained SELFIES embedding -> N in {0, 64, 256, 512}

This mirrors Ablation_study/ablation_performance.py's ABLATION_CONFIG for the
(bob, gaussian) and (selfies_ted, gaussian) combinations, but reuses Kermol's own
grid_search_cv_global / build_global_kernels / SpectralAnalyzer instead of duplicating
that logic.

Usage
-----
    python examples/ablation_study.py --rep both --data_path /path/to/qm9_data.npz \
        --rep_dir /path/to/Dataset/QM9
    python examples/ablation_study.py --rep bob --n_train 500 --n_test 500 \
        --data_path /path/to/qm9_data.npz

Output
------
    <workdir>/ablation_study.csv — one row per (rep, N, length_scale).
    <workdir>/ablation_study.png — test MAE vs N, one line per rep.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Kermol.data.features import load_precomputed_rep
from Kermol.kernels.kernels import build_global_kernels, grid_search_cv_global, krr_predict
from Kermol.kernels.spectral import SpectralAnalyzer

from full_kernel import LAMBDA_GRID, split_indices, generate_rep_split

# Per-rep N grids differ because bob (~30 features) and selfies_ted (1024 features)
# have very different feature budgets — see Ablation_study/ablation_performance.py.
ABLATION_CONFIG = {
    'bob':         {'n_values': [0, 16, 64],       'length_scales': [100., 1000., 10000.]},
    'selfies_ted': {'n_values': [0, 64, 256, 512], 'length_scales': [100., 1000., 10000.]},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rep', choices=['bob', 'selfies_ted', 'both'], default='both')
    p.add_argument('--property_name', default='gap', help='QM9 regression target.')
    p.add_argument('--n_train', type=int, default=1500)
    p.add_argument('--n_test', type=int, default=1500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cv_folds', type=int, default=4)
    p.add_argument('--data_path', required=True, help='Path to the QM9 .npz dataset.')
    p.add_argument('--rep_dir', default=None,
                   help="Directory containing selfies_ted.npy. Required when --rep is "
                        "'selfies_ted' or 'both'.")
    p.add_argument('--workdir', default=os.path.join(os.path.dirname(__file__), 'output'))
    args = p.parse_args()
    if args.rep in ('selfies_ted', 'both') and args.rep_dir is None:
        p.error("--rep_dir is required when --rep is 'selfies_ted' or 'both'")
    return args


def remove_features(X, n, seed):
    """Uniformly drop n random feature columns (no-op when n == 0)."""
    if n == 0:
        return X
    n = min(n, X.shape[1] - 1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[1], size=n, replace=False)
    return np.delete(X, idx, axis=1)


def load_rep(rep_name, coords, charges, train_idx, test_idx, rep_dir=None):
    if rep_name == 'bob':
        X_train, X_test = generate_rep_split('bob', coords, charges, train_idx, test_idx)
    else:
        X_full  = load_precomputed_rep('selfies_ted', rep_dir=rep_dir)
        X_train = X_full[train_idx]
        X_test  = X_full[test_idx]
    return np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)


def run_rep(rep_name, X_train_full, X_test_full, y_train, y_test, args):
    cfg  = ABLATION_CONFIG[rep_name]
    rows = []
    for n in cfg['n_values']:
        X_train = remove_features(X_train_full, n, seed=args.seed + n)
        X_test  = remove_features(X_test_full,  n, seed=args.seed + n)

        for ls in cfg['length_scales']:
            best = grid_search_cv_global(X_train, y_train, [ls], LAMBDA_GRID,
                                         'gaussian', cv=args.cv_folds, norm=2)
            K_train, K_test = build_global_kernels(X_train, X_test, ls, 'gaussian', norm=2)
            y_pred   = krr_predict(K_train, y_train, K_test, best['lambda'])
            test_mae = float(np.mean(np.abs(y_pred - y_test)))
            metrics  = SpectralAnalyzer(K_train).get_all_metrics(y=y_train, lam=best['lambda'])

            print(f"  {rep_name}  N={n:>4d}  ls={ls:<8.0f}  "
                  f"lambda={best['lambda']:.1e}  test_mae={test_mae:.4f}")
            rows.append({'rep': rep_name, 'N': n, 'length_scale': ls,
                        'lambda': best['lambda'], 'cv_mae': best['mae'],
                        'test_mae': test_mae, **metrics})
    return rows


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

    reps = ['bob', 'selfies_ted'] if args.rep == 'both' else [args.rep]

    all_rows = []
    for rep_name in reps:
        print(f"\nLoading '{rep_name}' representation …")
        X_train_full, X_test_full = load_rep(rep_name, coords, charges, train_idx, test_idx,
                                            rep_dir=args.rep_dir)
        print(f"  {rep_name}: {X_train_full.shape[1]} features")
        all_rows.extend(run_rep(rep_name, X_train_full, X_test_full, y_train, y_test, args))

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(args.workdir, 'ablation_study.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for rep_name in reps:
        sub = df[df['rep'] == rep_name]
        best_per_n = sub.groupby('N')['test_mae'].min().reset_index().sort_values('N')
        ax.plot(best_per_n['N'], best_per_n['test_mae'], marker='o', label=rep_name)
    ax.set_xlabel('N (features removed)')
    ax.set_ylabel(f'Test MAE ({args.property_name}, best length-scale)')
    ax.set_title('Ablation: feature removal vs test MAE')
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(args.workdir, 'ablation_study.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_png}")


if __name__ == '__main__':
    main()
