import os
import numpy as np
from typing import List, Union

# Pre-computed global representations stored as .npy files
PRECOMP_GLOBAL_REPS = {
    'slatm', 'grover_base', 'grover_large',
    'chembert', 'chemberta', 'selfies_ted', 'selformer',
}
FINGERPRINT_REPS = {'ecfp6', 'ecfp4'}


def generate_ecfp6(smiles: np.ndarray, radius: int = 3,
                   n_bits: int = 2048) -> np.ndarray:
    """Generate ECFP6 (Morgan radius-3) fingerprints from SMILES strings.

    Returns a float64 array of shape (n_mols, n_bits).
    Uses DataStructs.ConvertToNumpyArray to avoid returning an object array
    of RDKit BitVect objects, which cannot be converted to a torch tensor.
    """
    from rdkit.Chem import AllChem, MolFromSmiles
    from rdkit import DataStructs
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mols  = [MolFromSmiles(s) for s in smiles]
    fps   = np.zeros((len(mols), n_bits), dtype=np.float64)
    for i, mol in enumerate(mols):
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol), fps[i])
    return fps


def generate_ecfp4(smiles: np.ndarray, n_bits: int = 1024) -> np.ndarray:
    """Generate ECFP4 (Morgan radius-2, 1024 bits) fingerprints from SMILES strings."""
    from rdkit.Chem import AllChem, MolFromSmiles
    from rdkit import DataStructs
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
    mols  = [MolFromSmiles(s) for s in smiles]
    fps   = np.zeros((len(mols), n_bits), dtype=np.float64)
    for i, mol in enumerate(mols):
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol), fps[i])
    return fps


def load_smiles_from_csv(csv_path: str) -> np.ndarray:
    """Load SMILES strings from a CSV file with a 'SMILES' column."""
    import pandas as pd
    return pd.read_csv(csv_path, usecols=['SMILES'])['SMILES'].values
DEFAULT_REP_DIR = '/scratch/asmaj/Spec_Kern/Dataset'

def load_precomputed_rep(rep_name: str, rep_dir: str = DEFAULT_REP_DIR) -> np.ndarray:
    """
    Load a pre-computed global representation from a .npy file.

    Handles three storage formats:
      - plain float/int array  (grover_base, grover_large, chemberta, selfies_ted, selformer)
      - object array of fixed-size vectors (legacy format)
      - dict with rep_name key (legacy format)
    """
    path = os.path.join(rep_dir, f'{rep_name}.npy')
    data = np.load(path, allow_pickle=True)

    if data.dtype != object:
        return data.astype(np.float64)

    # Object array: try dict format first, then stack rows
    try:
        d = data.item()
        if isinstance(d, dict):
            return np.array(d[rep_name], dtype=np.float64)
    except (ValueError, KeyError):
        pass

    return np.vstack(data).astype(np.float64)

from rdkit.Chem import AllChem, MolFromSmiles
from ase import Atoms
from dscribe.descriptors import SOAP, ACSF
from qml2 import Compound, CompoundList
from qml2.representations.standard_geometric import array_ as qml_array
from qml2.representations import (
    generate_fchl19,
    generate_coulomb_matrix,
    get_slatm_mbtypes,
    generate_slatm,
    # New BOB imports
    compute_ncm, 
    get_bob_bags
)

# from qmllib.representations import generate_fchl19, generate_coulomb_matrix, generate_bob, get_slatm_mbtypes, generate_slatm, generate_acsf



class FingerprintGenerator:
    def __init__(self, radius: int = 3, n_bits: int = 2048):
        """Generates ECFP (Morgan) fingerprints using RDKit."""
        self.name = 'ecfp'
        self.radius = radius
        self.n_bits = n_bits

    def generate(self, smiles: Union[List[str], np.ndarray]) -> np.ndarray:
        """Generates Morgan (ECFP) fingerprints from SMILES."""
        
        fpgen = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        
        rdkit_mols = [MolFromSmiles(s) for s in smiles]
        fps = [fpgen.GetFingerprint(mol) for mol in rdkit_mols]
        
        return np.array(fps)


class PhysicalGenerator:
    """
    Physics-based representations.
    Supports both Global and Local (atomic) modes.
    """
    def __init__(self, representation: str, local: bool = False, **kwargs):
        """
        Args:
            representation: 'fchl19', 'soap', 'coulomb_matrix', 'bob', 'slatm', 'acsf'
            local: If True, returns atomic representations. If False, molecular.
            kwargs: representation-specific parameters (e.g. soap_species, interaction cuts).
        """
        self.name = representation
        self.representation = representation
        self.local = local
        self.kwargs = kwargs

    def generate(self, coords: np.ndarray, charges: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Args:
            coords: Array of coordinates (N_samples, ...)
            charges: Array of atomic numbers (N_samples, ...)
        """
        if self.representation == 'fchl19':
            return self._generate_fchl19(coords, charges)
        elif self.representation == 'soap':
            return self._generate_soap(coords, charges)
        elif self.representation == 'coulomb_matrix':
            return self._generate_cm(coords, charges)
        elif self.representation == 'bob':
            return self._generate_bob(coords, charges)
        elif self.representation == 'slatm':
            return self._generate_slatm(coords, charges)
        elif self.representation == 'acsf':
            return self._generate_acsf(coords, charges)
        else:
            raise ValueError(f"Unknown physical representation: {self.representation}")

    def _generate_fchl19(self, coords, charges):
        all_charges = np.concatenate(charges) if isinstance(charges[0], (list, np.ndarray)) else charges.flatten()
        unique_elements = np.unique(all_charges)
        
        reps = []
        for q, r in zip(charges, coords):
            reps.append(generate_fchl19(
                q, r, 
                elements=unique_elements, 
                nRs2=self.kwargs.get('nRs2', 240), 
                nRs3=self.kwargs.get('nRs3', 308)
            ))
        return reps

    def _generate_soap(self, coords, charges):
        species = self.kwargs.get('species', ["H", "C", "O", "N", "F"])
        soap = SOAP(
            species=species,
            periodic=False,
            r_cut=self.kwargs.get('r_cut', 6.0),
            n_max=self.kwargs.get('n_max', 3),
            l_max=self.kwargs.get('l_max', 3),
            sigma=self.kwargs.get('sigma', 0.1),  # paper uses 0.1; DScribe default is 1.0
        )
        reps = []
        for q, r in zip(charges, coords):
            mol = Atoms(numbers=q, positions=r)
            reps.append(soap.create(mol))
        return reps

    def _generate_cm(self, coords, charges):
        max_atoms = max(len(q) for q in charges)
        size = max_atoms        # paper uses size=29 → 29*30/2 = 435 features
        reps = []
        for q, r in zip(charges, coords):
            q_arr = np.array(q, dtype=np.int32)    
            r_arr = np.array(r, dtype=np.float64)
            reps.append(generate_coulomb_matrix(q_arr, r_arr, size=size))
        return np.array(reps)

    def _generate_bob(self, coords, charges):
        compounds = [
            Compound(
                coordinates=np.array(xyz, dtype=np.float64), 
                nuclear_charges=np.array(Z, dtype=np.int32)
            )
            for xyz, Z in zip(coords, charges)
        ]
        compound_list = CompoundList(compounds)

        elements = qml_array([1,6,7,8,9])

        bags = get_bob_bags(compound_list.all_nuclear_charges(), elements=elements)
        ncm = compute_ncm(bags)

        compound_list.generate_bob(
            bags,
            ncm=ncm,
            elements=elements,
            test_mode=True 
        )


        rep = np.array(compound_list.all_representations())
        return rep

    def _generate_slatm(self, coords, charges):
        mbtypes = get_slatm_mbtypes([q for q in charges])
        reps = []
        for q, r in zip(charges, coords):
            reps.append(generate_slatm(q, r, mbtypes=mbtypes, local=self.local))
        return np.array(reps)


    def _padding_atomic_matrix(self, mat,pad):
        temp = np.zeros((pad,mat.shape[-1]))
        size = mat.shape[0]
        temp[:size] = mat
        return temp

    def _generate_acsf(self, coords, charges):
        species = np.unique(np.concatenate(charges)).astype(np.int64)
        pad = max([len(arr) for arr in charges])
        mols = [Atoms(positions=r,numbers=q) for r,q in zip(coords,charges)]
        acsf = ACSF(
            species=species,
            r_cut=6.0,
            g2_params=[[1, 1], [1, 2], [1, 3]],
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
            )
        repacsf = []
        for j in range(len(charges)):
            rep = acsf.create(mols[j],n_jobs=36)
            repacsf.append(self._padding_atomic_matrix(rep,pad))
        repacsf = np.array(repacsf)
        return repacsf
