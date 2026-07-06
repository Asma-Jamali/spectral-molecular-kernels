"""
Download, clean, and package the QM9 dataset.
"""

import glob
import os
import tarfile
import urllib.request

import numpy as np
from tqdm import tqdm
from qml2.dataset_formats.qm9 import Quantity, read_SMILES, atomization_energy

qm9_download_url = "https://figshare.com/ndownloader/files/3195389"
qm9_archive_basename = "dsgdb9nsd.xyz.tar.bz2"

_PERIODIC = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

_QUANTITIES = {
    "gap": Quantity("HOMO-LUMO gap"),
    "U0": Quantity("Internal energy at 0 K"),
    "U298": Quantity("Internal energy at 298.15 K"),
    "H298": Quantity("Enthalpy at 298.15 K"),
    "G298": Quantity("Free energy at 298.15 K"),
    "Cv": Quantity("Heat capacity at 298.15 K"),
    "ZPVE": Quantity("Zero point vibrational energy"),
}


def download(dest: str = qm9_archive_basename) -> None:
    print(f"Downloading QM9 from figshare …")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  {pct:5.1f}%  ({downloaded // 1_000_000} / {total_size // 1_000_000} MB)",
                  end='', flush=True)

    urllib.request.urlretrieve(qm9_download_url, dest, reporthook=_progress)
    print()


def extract(archive: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive, 'r:bz2') as tar:
        tar.extractall(dest_dir)


def fix_file(path: str) -> None:
    """Fix broken scientific notation patterns in a single QM9 xyz file."""
    with open(path) as f:
        text = f.read()
    text = (text
            .replace("*^", "e")
            .replace("*-", "e-")
            .replace("*+", "e+")
            .replace("*10^", "e"))
    with open(path, 'w') as f:
        f.write(text)


def fix_all(xyz_dir: str) -> None:
    """Fix broken scientific notation in all xyz files in xyz_dir."""
    files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))
    for f in tqdm(files, desc="Fixing notation"):
        fix_file(f)


def _read_xyz(path: str):
    """Return (atom_symbols, coords_array, atomic_numbers) from one xyz file."""
    with open(path) as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    atoms, coords = [], []
    for line in lines[2: 2 + natoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append(list(map(float, parts[1:4])))
    return atoms, np.array(coords), [_PERIODIC[a] for a in atoms]


def load_dataset(xyz_dir: str) -> list:
    """Parse all QM9 xyz files in xyz_dir and return a list of record dicts."""
    files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))
    print(f"Found {len(files)} xyz files in {xyz_dir}/")
    records = []
    for f in tqdm(files, desc="Parsing"):
        try:
            atoms, coords, Z = _read_xyz(f)
            record = {
                "filename": f,
                "smiles": read_SMILES(f),
                "atoms": atoms,
                "coords": coords,
                "Z": Z,
                "AtomizationEnergy": atomization_energy(f),
            }
            for key, q in _QUANTITIES.items():
                record[key] = q.extract_xyz(f)
            records.append(record)
        except Exception as e:
            print(f" skipping {os.path.basename(f)}: {e}")
    return records


def save_npz(records: list, out_path: str) -> None:
    """Save a list of QM9 record dicts to a .npz file."""
    keys_obj = ["filename", "smiles", "atoms", "coords", "Z"]
    keys_float = ["gap", "U0", "U298", "H298", "G298", "Cv", "ZPVE", "AtomizationEnergy"]
    arrays = {}
    for k in keys_obj:
        arrays[k] = np.array([r[k] for r in records], dtype=object)
    for k in keys_float:
        arrays[k] = np.array([r[k] for r in records])
    np.savez(out_path, **arrays)



def prepare(xyz_dir: str = "qm9_xyz",
            archive: str = qm9_archive_basename,
            out_path: str = "qm9_data.npz") -> None:
    if not os.path.exists(archive):
        download(archive)
    else:
        print(f"Archive already present: {archive}")

    if not glob.glob(os.path.join(xyz_dir, "*.xyz")):
        extract(archive, xyz_dir)
    else:
        print(f"xyz files already in {xyz_dir}/")

    fix_all(xyz_dir)
    records = load_dataset(xyz_dir)
    save_npz(records, out_path)
