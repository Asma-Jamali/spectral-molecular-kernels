import numpy as np
import pandas as pd
from rdkit import Chem

HARTREE_PROPS = {'U0', 'U298', 'H298', 'G298', 'ZPVE', 'gap', 'homo', 'lumo'}
ENERGY_PROPS  = {'U0', 'U298', 'H298', 'G298'}   # subset that supports atomization
RAW_PROPS     = {'Cv'}

PROP_UNITS = {
    'U0': 'eV', 'U298': 'eV', 'H298': 'eV', 'G298': 'eV', 'ZPVE': 'eV',
    'zpve': 'eV', 'gap': 'eV', 'homo': 'eV', 'lumo': 'eV',
    'Cv': 'cal/(mol·K)',
}

# Maps property_name → column name in filtered_QM9.csv
CSV_COL_MAP = {
    'U0': 'U0', 'U298': 'U298', 'H298': 'H298', 'G298': 'G298',
    'ZPVE': 'ZPVE', 'gap': 'gap', 'homo': 'HOMO', 'lumo': 'LUMO', 'Cv': 'Cv',
}

# Atomic reference energies at B3LYP/6-31G(2df,p) in Hartree
# Source: Ramakrishnan et al., Sci. Data 1, 140022 (2014), Table S1
ATOMIC_REFS = {
    'U0':   {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730},
    'U298': {1: -0.498857, 6: -37.845355, 7: -54.582445, 8: -75.063163, 9: -99.717314},
    'H298': {1: -0.497912, 6: -37.844411, 7: -54.581501, 8: -75.062219, 9: -99.716370},
    'G298': {1: -0.510927, 6: -37.861317, 7: -54.598897, 8: -75.079532, 9: -99.733544},
}

HARTREE_TO_EV = 27.2114


def atomization_energies(raw_ha, charges, prop):
    """Subtract per-atom B3LYP references and convert Hartree → eV."""
    refs    = ATOMIC_REFS[prop]
    ref_sum = np.array([sum(refs[int(z)] for z in mol_charges if z > 0)
                        for mol_charges in charges])
    return (ref_sum - raw_ha) * HARTREE_TO_EV


def charges_from_smiles(smiles: np.ndarray) -> np.ndarray:
    """
    Derive per-molecule atomic-number arrays from SMILES strings.
    Explicit hydrogens are added so the atomic reference sums match QM9.
    Returns an object array of int32 arrays, same format as npz['Z'].
    """
    charges = np.empty(len(smiles), dtype=object)
    for i, smi in enumerate(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        charges[i] = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    return charges


def prepare_labels_from_csv(csv_path: str, property_name: str,
                            use_atomization: bool = False,
                            charges: np.ndarray | None = None):
    """
    Load labels from filtered_QM9.csv.
    Returns (labels, train_unit, prop_unit) with the same semantics as prepare_labels.
    use_atomization requires charges (atomic numbers array) from the .npz file.
    """
    need_smiles = use_atomization and property_name in ENERGY_PROPS and charges is None
    cols = [CSV_COL_MAP.get(property_name, property_name)]
    if need_smiles:
        cols.append('SMILES')

    df  = pd.read_csv(csv_path, usecols=cols)
    raw = df[CSV_COL_MAP.get(property_name, property_name)].values.astype(float)

    prop_unit = PROP_UNITS.get(property_name, '')

    if property_name in HARTREE_PROPS:
        if use_atomization and property_name in ENERGY_PROPS:
            if charges is None:
                charges = charges_from_smiles(df['SMILES'].values)
            labels = atomization_energies(raw, charges, property_name)
        else:
            labels = raw * HARTREE_TO_EV
        train_unit = 'eV'
    else:
        labels     = raw
        train_unit = prop_unit

    return labels, train_unit, prop_unit


def prepare_labels(data, property_name, charges, use_atomization):
    """
    Returns (labels, train_unit, prop_unit).
    All Hartree-based properties are converted to eV before training.
    Cv is left in cal/(mol·K). train_unit reflects the unit of labels.
    """
    prop_unit = PROP_UNITS.get(property_name, '')

    if property_name in HARTREE_PROPS:
        if use_atomization and property_name in ENERGY_PROPS:
            labels = atomization_energies(data[property_name], charges, property_name)
        else:
            labels = data[property_name].astype(float) * HARTREE_TO_EV
        train_unit = 'eV'
    else:
        labels     = data[property_name].astype(float)
        train_unit = prop_unit

    return labels, train_unit, prop_unit
