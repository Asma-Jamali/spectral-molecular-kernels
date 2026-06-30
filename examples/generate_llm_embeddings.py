"""
generate_llm_embeddings.py — Pre-compute pretrained-LLM molecular embeddings
=============================================================================
Runs one of the registered SELFIES-based transformer models
(Kermol.data.features.LLM_MODEL_REGISTRY: chembert, chemberta, selformer,
selfies_ted) over every SMILES in a dataset and saves the resulting
(n_molecules, hidden_size) embedding matrix as a .npy file, ready to be
loaded with Kermol.data.features.load_precomputed_rep.

Usage
-----
    python examples/generate_llm_embeddings.py --model chembert \
        --data_path /path/to/qm9_data.npz
    python examples/generate_llm_embeddings.py --model selfies_ted \
        --data_path /path/to/qm9_data.npz --out /path/to/selfies_ted.npy
"""

import argparse
import os

import numpy as np

from Kermol.data.features import LLMEmbeddingGenerator, LLM_MODEL_REGISTRY

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, choices=sorted(LLM_MODEL_REGISTRY),
                        help='Which registered pretrained model to run.')
    parser.add_argument('--data_path', required=True,
                        help='Path to .npz dataset with a "smiles" key.')
    parser.add_argument('--out', default=None,
                        help='Output .npy path. Defaults to Dataset/QM9/<model>.npy')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = args.out or os.path.join(REPO_ROOT, 'Dataset', 'QM9', f'{args.model}.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading SMILES from {args.data_path} …")
    data   = np.load(args.data_path, allow_pickle=True)
    smiles = data['smiles']
    print(f"  {len(smiles)} molecules")

    cfg = LLM_MODEL_REGISTRY[args.model]
    print(f"Loading model '{args.model}' ({cfg['hf_id']}) …")
    gen = LLMEmbeddingGenerator(args.model, batch_size=args.batch_size,
                                max_length=args.max_length, device=args.device)

    print("Generating embeddings …")
    embeddings = gen.generate(smiles)
    print(f"  shape: {embeddings.shape}")

    np.save(out_path, embeddings)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
