"""
Builds data/operators/<folder>/ from the xainano/handwrittenmathsymbols
Kaggle dataset (via kagglehub) -- covers every operator we need from one
source, one folder per symbol under extracted_images/.

Note: the archive is a .rar, and kagglehub's extraction can silently stall
partway through on machines without unrar/7-Zip, leaving some symbol
folders missing. If this script reports fewer than 6 folders found, install
7-Zip and re-run -- it only copies what kagglehub already extracted, it
doesn't extract the archive itself.

Usage:
    pip install kagglehub
    python data/prepare_kaggle_operators.py
"""
import os
import shutil

import kagglehub

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operators")

# Kaggle extracted_images/ folder name -> our output folder name
SYMBOL_FOLDERS = {
    "+": "plus",
    "-": "minus",
    "times": "times",
    "div": "div",
    "(": "lparen",
    ")": "rparen",
}


def main(max_per_symbol: int = 300):
    dataset_path = kagglehub.dataset_download("xainano/handwrittenmathsymbols")
    images_root = os.path.join(dataset_path, "extracted_images")

    if not os.path.isdir(images_root):
        raise FileNotFoundError(
            f"Couldn't find {images_root} -- kagglehub's download/extraction "
            "may have changed layout; check the dataset page for the current "
            "folder structure."
        )

    found = 0
    for kaggle_name, out_name in SYMBOL_FOLDERS.items():
        src_dir = os.path.join(images_root, kaggle_name)
        if not os.path.isdir(src_dir):
            print(f"WARNING: {src_dir} not found, skipping {out_name}")
            continue

        out_dir = os.path.join(OUT_ROOT, out_name)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(os.listdir(src_dir))[:max_per_symbol]
        for i, fname in enumerate(files):
            shutil.copyfile(os.path.join(src_dir, fname),
                             os.path.join(out_dir, f"{i:04d}.png"))
        print(f"{out_name}: {len(files)} images")
        found += 1

    if found < len(SYMBOL_FOLDERS):
        print(
            f"\nOnly found {found}/{len(SYMBOL_FOLDERS)} symbol folders under "
            f"{images_root} -- the dataset should have ~80 folders total, so "
            "kagglehub probably couldn't fully extract the .rar. Install "
            "7-Zip or unrar, re-download, and re-run this script."
        )


if __name__ == "__main__":
    main()
