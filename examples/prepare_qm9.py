import argparse
from Kermol.data.qm9 import prepare


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='qm9_data.npz',
                   help='Output .npz path (default: qm9_data.npz).')
    p.add_argument('--xyz_dir', default='qm9_xyz',
                   help='Directory to extract xyz files into (default: qm9_xyz/).')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prepare(xyz_dir=args.xyz_dir, out_path=args.out)
