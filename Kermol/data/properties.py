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

CSV_COL_MAP = {
    'U0': 'U0', 'U298': 'U298', 'H298': 'H298', 'G298': 'G298',
    'ZPVE': 'ZPVE', 'gap': 'gap', 'homo': 'HOMO', 'lumo': 'LUMO', 'Cv': 'Cv',
}

atom_energies = {
    "H": -0.500273,
    "C": -37.846772,
    "N": -54.583861,
    "O": -75.064579,
    "F": -99.718730,
}

ATOMIC_NUM_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

HARTREE_TO_EV = 27.2114


def atomization_energies(raw_ha, charges, prop):
    """Subtract per-atom B3LYP references and convert Hartree to eV."""
    ref_sum = np.array([sum(atom_energies[ATOMIC_NUM_TO_SYMBOL[int(z)]] for z in mol_charges if z > 0)
                        for mol_charges in charges])
    return (ref_sum - raw_ha) * HARTREE_TO_EV


def charges_from_smiles(smiles: np.ndarray) -> np.ndarray:
    charges = np.empty(len(smiles), dtype=object)
    for i, smi in enumerate(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        charges[i] = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    return charges

def prepare_labels_from_csv(csv_path: str, property_name: str,
                            use_atomization: bool = False,
                            charges: np.ndarray | None = None):
    
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
