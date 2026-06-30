# Spectral Analysis of Molecular Kernel Methods

Kernel-ridge regression (KRR) on QM9 molecular properties, with a focus on the **spectral behaviour** of the kernel matrix itself: how its eigenspectrum relates to test error, how much of that spectrum is actually needed (truncation), and how robust different molecular representations are to feature removal (ablation).

## Representations and kernels

| Category | Names | Notes |
|---|---|---|
| Global 3D | `coulomb_matrix`, `bob`, `slatm`, `fchl19` | Use `build_global_kernels` — Gaussian/Laplacian/linear via pairwise distance |
| Local 3D | `soap`, `acsf`, `fchl19` (local mode) | Use `build_local_kernels` ([qml2](https://github.com/qml2code/qml2)). Pass nuclear charges (`Z_tr`/`Z_te`) to get the Kronecker-delta ("dn") variant, which only lets same-element atom pairs contribute — the physically correct choice for heteronuclear molecules |
| Fingerprint | `ecfp4`, `ecfp6` | Morgan fingerprints from SMILES (RDKit); paired with kernels introduced in [GAUCHE](https://github.com/leojklarner/gauche) (`tanimoto`, `dice`, …) or with gaussian/laplacian via `build_global_kernels` |
| Transformer-based  | `selfies_ted`, `grover_base`, `grover_large`, `chembert`, `chemberta`, `selformer` | **Load** any of them with `load_precomputed_rep('<name>', rep_dir='/path/to/Dataset/QM9')` → `(n_mols, hidden_dim)` float32 array, used as a global vector representation. |

`LLMEmbeddingGenerator` (in `Kermol.data.features`) converts each SMILES to a
space-tokenized SELFIES string, runs it through a pretrained HuggingFace model, and
pools the output to a single vector — CLS/`pooler_output` for `chembert`/`chemberta`,
attention-mask-weighted mean for `selformer`/`selfies_ted`.

## Spectral analysis

Given `K_train`, decomposed as $K = \sum_i \mu_i u_i u_i^\top$ (eigenvalues $\mu_1 \geq \mu_2 \geq \dots$, eigenvectors $u_i$), `Kermol.kernels.spectral.SpectralAnalyzer` computes:

| Metric | Key | Equation |
|---|---|---|
| Spectral Shannon entropy | `SSE` | $\exp\left(-\sum_i p_i \log p_i\right), \quad p_i = \dfrac{\mu_i}{\sum_j \mu_j}$ |
| Intrinsic dimension | `ID` | $\dfrac{\sum_i \mu_i}{\mu_1}$ |
| Stable rank | `SR` | $\dfrac{\sum_i \mu_i^2}{\mu_1^2}$ |
| Power-law decay exponent | `alpha` | $\mu_i \sim i^{-\alpha}$ (fit on the tail of the spectrum) |
| Target-weighted effective dimension | `tw_eff` | $\sum_i \dfrac{\mu_i}{\mu_i + \lambda} \cdot \dfrac{(u_i^\top y)^2}{\lVert y \rVert^2}$ |

## Examples

| Script | Description | Key inputs |
|---|---|---|
| `full_kernel.py` | Builds a Gaussian kernel, fits KRR, and reports test MAE/R² and spectral metrics of K_train | `--mode {global,local}`, `--data_path`, `--sigma`, `--lam`, `--property_name`, `--n_train`, `--n_test` |
| `truncated_kernel.py` | Same as above, then sweeps truncation rank k to show how error grows as kernel rank drops | `--mode {global,local}`, `--data_path`, `--n_train`, `--n_test`, `--n_val` |
| `ablation_study.py` | Randomly removes N features from `bob` or `selfies_ted` and re-fits, sweeping over N | `--rep {bob,selfies_ted,both}`, `--data_path`, `--rep_dir` (required for `selfies_ted`), `--n_train`, `--n_test` |
| `generate_llm_embeddings.py` | Generates a transformer embedding matrix for a dataset and saves it as `.npy` | `--model {chembert,chemberta,selformer,selfies_ted}`, `--data_path`, `--out`, `--device` |

## Setup

```bash
git clone https://github.com/Asma-Jamali/spectral-molecular-kernels.git
cd spectral-molecular-kernels
```

Then, install all dependencies and register `Kermol` as an importable package.
  ```bash
  pip install -r requirements.txt
  pip install -e . --no-deps
  ```
