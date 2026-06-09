"""
Optional offline preprocessing: convert raw NIfTI volumes to numpy arrays
so the DataLoader doesn't re-process on every epoch.

Usage:
    python scripts/preprocess.py --data-root data/raw --out-dir data/processed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.data.loader import build_index
from src.preprocessing.transforms import preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",    default="data/raw")
    parser.add_argument("--out-dir",      default="data/processed")
    parser.add_argument("--skull-strip",  action="store_true",
                        help="Zero out extracranial voxels before normalization")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "volumes").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    from src.data.loader import load_nifti
    records = build_index(args.data_root)

    for i, r in enumerate(tqdm(records, desc="Preprocessing")):
        volume, _ = load_nifti(r["image"])
        mask,   _ = load_nifti(r["mask"])
        vol_pp, msk_pp = preprocess(volume, mask, skull_strip=args.skull_strip)

        stem = Path(r["image"]).name.split(".")[0]
        np.save(out_dir / "volumes" / f"{stem}.npy", vol_pp)
        np.save(out_dir / "masks"   / f"{stem}.npy", msk_pp)

    print(f"Saved {len(records)} preprocessed pairs to {out_dir}")


if __name__ == "__main__":
    main()
