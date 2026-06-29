import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

from .data.properties import ENERGY_PROPS, HARTREE_TO_EV

PROP_LABELS = {
    'U0': 'U$_0$', 'U298': 'U$_{298}$', 'H298': 'H$_{298}$', 'G298': 'G$_{298}$',
    'zpve': 'ZPVE', 'gap': 'HOMO-LUMO gap', 'Cv': '$C_v$',
}


def save_learning_curve(results, train_sizes, tag, property_name, kernel_name,
                        use_atomization, train_unit, prop_unit, props_dir):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(props_dir, f'learning_curve_{tag}_raw.csv'), index=False)

    summary = (df.groupby('n_train')['test_mae']
               .agg(mean_mae='mean', std_mae='std')
               .reset_index())
    summary.to_csv(os.path.join(props_dir, f'learning_curve_{tag}.csv'), index=False)
    print(f'\nLearning-curve results saved to Properties/learning_curve_{tag}*.csv')
    print(summary.to_string(index=False))

    title_suffix = 'atomization' if (property_name in ENERGY_PROPS and use_atomization) else 'QM9'
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(summary['n_train'], summary['mean_mae'], yerr=summary['std_mae'],
                marker='o', linewidth=1.5, capsize=4, label='Test MAE (mean ± std)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel(f'MAE ({prop_unit})')
    ax.set_title(f'Learning curve — {kernel_name} KRR on QM9 '
                 f'({PROP_LABELS.get(property_name, property_name)}, {title_suffix})')
    ax.set_xticks(train_sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(props_dir, f'learning_curve_{tag}.png'), dpi=150)
    plt.close(fig)
    print(f'Plot saved to Properties/learning_curve_{tag}.png')


def save_spectral_results(spectral_rows, eigenvalue_store, tag, spectrum_dir):
    spec_df = pd.DataFrame(spectral_rows)
    spec_df.to_csv(os.path.join(spectrum_dir, f'spectral_metrics_{tag}_raw.csv'), index=False)

    spec_summary = (spec_df.groupby('n_train')[['SSE', 'ID', 'SR', 'alpha', 'alpha_r2', 'tw_eff']]
                    .agg(['mean', 'std'])
                    .reset_index())
    spec_summary.columns = ['_'.join(c).strip('_') for c in spec_summary.columns]
    spec_summary.to_csv(os.path.join(spectrum_dir, f'spectral_metrics_{tag}.csv'), index=False)
    print(f'Spectral metrics saved to Spectrum/spectral_metrics_{tag}*.csv')

    eigval_path = os.path.join(spectrum_dir, f'eigenvalues_{tag}.npz')
    np.savez(eigval_path, **{f's{s}_n{n}': v for (s, n), v in eigenvalue_store.items()})
    print(f'Eigenvalues saved to Spectrum/eigenvalues_{tag}.npz')
