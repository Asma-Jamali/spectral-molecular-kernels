"""
submit_tr.py — SLURM job submitter for the spectral truncation study
=====================================================================
Mirrors submit_jobs.py but targets main_truncate.py.

Edit the experiment config block below, then run:
    python submit_tr.py
"""

import os
import time

# ── Default grids (mirror submit_jobs.py conventions) ────────────────────────
LOCAL_LENGTH_GRID  = [0.1 * (2 ** n) for n in range(15)]    # 0.1 → 1638.4
GLOBAL_LENGTH_GRID = [10 ** n for n in range(2, 9)]          # 1e2 → 1e8
LAMBDA_GRID        = [10 ** (-3 * n) for n in range(1, 10)]  # 1e-3 → 1e-27
TRUNC_LAMBDA_GRID  = [10 ** (-n) for n in range(3, 16)]      # 1e-3 → 1e-15
FP_LAMBDA_GRID     = [10 ** i for i in range(-10, 3)]        # 1e-10 → 1e2

LOCAL_REPS        = {'soap', 'fchl19', 'acsf'}
FINGERPRINT_REPS  = {'ecfp6', 'ecfp4'}
FINGERPRINT_KERNELS = {
    'tanimoto', 'dice', 'otsuka', 'sogenfrei', 'braunblanquet',
    'faith', 'forbes', 'innerproduct', 'intersection', 'min_max', 'rand',
    'gaussian', 'laplacian',
}
ECFP6_KERNELS = sorted(FINGERPRINT_KERNELS)


def sh_file(
    workdir, data_path, property_name, kernel_name, rep_name,
    n_train, n_test, n_val, seeds, cv_folds, norm,
    length_grid, lambda_grid, trunc_lambda_grid,
    use_atomization=False, csv_path=None, rep_dir=None,
    warmup_sizes=None,
):
    """Generate and immediately submit a SLURM script."""
    
    if use_atomization:
        property_name += '_atm' # suffix to distinguish from raw property runs

    log_dir     = os.path.join(workdir, f'slurm_logs/Truncation/{property_name}/{kernel_name}')
    os.makedirs(log_dir, exist_ok=True)

    job_id      = f"{rep_name}"
    script_path = os.path.join(log_dir, f"submit_{job_id}.sh")
    log_path    = os.path.join(log_dir, f"{job_id}.out")

    # Memory: K_train (200 MB) + K_test (400 MB) + eigvecs (200 MB) + overhead
    if rep_name in LOCAL_REPS:
        mem        = "64G"
        time_limit = "10:00:00"
    else:
        mem        = "32G"
        time_limit = "3:00:00"

    length_str       = " ".join(str(v) for v in length_grid)
    lambda_str       = " ".join(str(v) for v in lambda_grid)
    trunc_lambda_str = " ".join(str(v) for v in trunc_lambda_grid)
    seeds_str        = " ".join(str(s) for s in seeds)

    warmup_str = (" --warmup_sizes " + " ".join(str(s) for s in sorted(warmup_sizes))
                  if warmup_sizes else "")

    cmd = (
        f"python {os.path.join(workdir, 'Tuning/main_truncate.py')} "
        + (f"--data_path {data_path} " if data_path else "")
        + (f"--csv_path {csv_path} "   if csv_path  else "")
        + (f"--rep_dir {rep_dir} "     if rep_dir   else "")
        + f"--workdir {workdir} "
        f"--property_name {property_name} "
        f"--rep_name {rep_name} "
        f"--kernel_name {kernel_name} "
        f"--n_train {n_train} "
        f"--n_test {n_test} "
        f"--n_val {n_val} "
        f"--seeds {seeds_str} "
        f"--cv_folds {cv_folds} "
        f"--norm {norm} "
        f"--length_grid {length_str} "
        f"--lambda_grid {lambda_str} "
        f"--trunc_lambda_grid {trunc_lambda_str}"
        + warmup_str
        + (" --use_atomization" if use_atomization else "")
    )

    with open(script_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --account=def-ravh011\n')
        f.write(f'#SBATCH --job-name={job_id}\n')
        f.write(f'#SBATCH --output={log_path}\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --cpus-per-task=10\n')
        f.write(f'#SBATCH --mem={mem}\n')
        f.write(f'#SBATCH --time={time_limit}\n\n')
        f.write('# --- Environment ---\n')
        f.write('module load python/3.12\n')
        f.write('source $HOME/virtual_envs/Spec_anal/bin/activate\n')
        f.write(f'export PYTHONPATH=$PYTHONPATH:{workdir}\n\n')
        f.write('# --- Run ---\n')
        f.write(cmd + '\n')

    print(f"Submitting: {job_id}")
    os.system(f"sbatch {script_path}")
    time.sleep(0.5)


def main():
    workdir = os.getcwd()

    # ── Experiment config ─────────────────────────────────────────────────────
    # Set data_path for on-the-fly reps (coulomb_matrix, bob, …)
    # Set csv_path  for pre-computed reps (slatm, grover_base, chembert, …)
    data_path = '/scratch/asmaj/spectral-molecular-kernels/dataset/qm9_data.npz'
    csv_path       = '/scratch/asmaj/Molkern/Dataset/QM9_reps/filtered_QM9.csv'  # kept for future use
    local_rep_dir  = '/scratch/asmaj/Spec_Kern/Dataset'
    precomp_rep_dir= '/scratch/asmaj/Spec_Kern/Dataset'

    property_names = ['Cv','ZPVE', 'U0', 'gap', 'H298', 'G298', 'U298']        # e.g. 'U0', 'gap', 'homo', 'lumo', 'Cv'
    kernel_list    = ['gaussian', 'laplacian']#, 'linear']  # 'gaussian', 'laplacian'
    rep_list       = ['ecfp4']#['chemberta', 'selfies_ted', 'selformer']#, 'slatm', 'grover_base', 'grover_large', 'chembert', 'selfies_ted', 'selformer', 'acsf', 'soap', 'fchl19']
    # Pre-computed global reps: 'slatm', 'grover_base', 'grover_large',
    #                            'chembert', 'selfies_ted', 'selformer'

    n_train         = 5000
    n_test          = 10_000
    n_val           = 1000
    seeds           = [42, 346, 90867, 12345]
    cv_folds        = 4
    use_atomization = False

    # Reference train-size sequence used in submit_jobs.py / train.py.
    # Sizes smaller than n_train are drawn first (and discarded) so the RNG
    # reaches the same state as the learning-curve run at n_train.
    TRAIN_SIZES_REF = [500, 1000, 2000, 5000, 10000, 20000]
    warmup_sizes    = [s for s in TRAIN_SIZES_REF if s < n_train]
    # ─────────────────────────────────────────────────────────────────────────

    PRECOMP   = {'slatm', 'grover_base', 'grover_large', 'chembert', 'chemberta', 'selfies_ted', 'selformer'}
    CSV_REPS  = set()  # all reps use npz for labels; set to {'chembert','selfies_ted','selformer'} to re-enable CSV

    for prop in property_names:
        for rep in rep_list:
            is_local       = rep.lower() in LOCAL_REPS
            is_precomp     = rep.lower() in PRECOMP
            is_fingerprint = rep.lower() in FINGERPRINT_REPS
            kernels        = ECFP6_KERNELS if is_fingerprint else kernel_list

            for kernel in kernels:
                is_fp_kernel = kernel in FINGERPRINT_KERNELS

                # linear kernel not supported for local reps
                if kernel == 'linear' and is_local:
                    continue

                length_grid = LOCAL_LENGTH_GRID if is_local else GLOBAL_LENGTH_GRID
                norm        = 1 if kernel == 'laplacian' else 2
                if is_local:
                    _rep_dir = local_rep_dir
                elif is_precomp:
                    _rep_dir = precomp_rep_dir
                else:
                    _rep_dir = None

                use_csv = rep.lower() in CSV_REPS
                sh_file(
                    workdir           = workdir,
                    data_path         = None if use_csv else data_path,
                    csv_path          = csv_path if use_csv else None,
                    rep_dir           = _rep_dir,
                    property_name     = prop,
                    kernel_name       = kernel,
                    rep_name          = rep,
                    n_train           = n_train,
                    n_test            = n_test,
                    n_val             = n_val,
                    seeds              = seeds,
                    cv_folds          = cv_folds,
                    norm              = norm,
                    length_grid       = length_grid,
                    lambda_grid       = FP_LAMBDA_GRID if (is_fingerprint and is_fp_kernel) else LAMBDA_GRID,
                    trunc_lambda_grid = FP_LAMBDA_GRID if (is_fingerprint and is_fp_kernel) else TRUNC_LAMBDA_GRID,
                    use_atomization   = use_atomization,
                    warmup_sizes      = warmup_sizes,
                )


if __name__ == '__main__':
    main()
