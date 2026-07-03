"""
generate pretrained-LLM molecular embeddings
Cover MLT-BERT, CHEMBERTa, SELFormer, SELFIESTED, and other registered models in Kermol.data.features.LLM_MODEL_REGISTRY.
"""

import argparse
import os

import numpy as np

from Kermol.data.features import LLMEmbeddingGenerator, LLM_MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=sorted(LLM_MODEL_REGISTRY), help='Which registered pretrained model to run.')
    parser.add_argument('--data_path', required=True, help='Path to .npz dataset with a "smiles" key.')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = f"Embeddings/{args.model}_embeddings.npy"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = np.load(args.data_path, allow_pickle=True)
    smiles = data['smiles']
    print(f" {len(smiles)} molecules")

    cfg = LLM_MODEL_REGISTRY[args.model]
    print(f"Loading model '{args.model}' ({cfg['hf_id']}) …")
    gen = LLMEmbeddingGenerator(args.model, batch_size=args.batch_size,
                                max_length=args.max_length, device=args.device)

    print("Generating embeddings …")
    embeddings = gen.generate(smiles)
    print(f"  shape: {embeddings.shape}")
    np.save(output_path, embeddings)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
