import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Union, Optional
import os

class QM9Loader:
    """
    Loading of QM9 data from .npz file.
    Expects keys atoms, SMILES, coords, Z (atomic numbers), gap (HUMO-LOMO Gap), U0 (Internal energy at 0K),
    U298 (internal energy at 298K), H298 (Enthalpy at 298K), G298 (Gibbs energy at 298K),
    Cv (Heat capacity), ZPVE (Zero-point vibrational energy )).
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self._data = None

    def _load_npz(self):
        """Load the npz file."""
        self._data = np.load(self.data_path, allow_pickle=True)


    def get_property(self, property_name: str) -> np.ndarray:
        """Retrieves a specific regression target (label)."""
        self._load_npz()
        if property_name not in self._data:
            raise KeyError(f"Property '{property_name}' not found in dataset.")
        return self._data[property_name]

    def get_smiles(self) -> np.ndarray:
        """Retrieves SMILES strings for fingerprint generation."""
        self._load_npz()
        if 'SMILES' not in self._data:
            raise KeyError("Key 'SMILES' missing from .npz file.")
        return self._data['SMILES']

    def get_structures(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves physical structure data.
        Returns:
            coords (np.ndarray): Shape (N, Max_Atoms, 3) or object array.
            charges (np.ndarray): Shape (N, Max_Atoms) or object array.
        """
        self._load_npz()
        # Adjust keys 'R' and 'Z' based on your specific .npz generation method
        if 'coords' not in self._data or 'Z' not in self._data:
             raise KeyError("Keys 'coords' (coordinates) or 'Z' (charges) missing.")
        return self._data['coords'], self._data['Z']