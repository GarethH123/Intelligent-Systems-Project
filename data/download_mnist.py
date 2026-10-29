"""
Downloads the MNIST digits as CSV (label, then 784 pixels per row) into
data/mnist_train.csv and data/mnist_test.csv -- the files data_loader.py reads.
Uses the oddrationale/mnist-in-csv Kaggle dataset via kagglehub.

Usage:
    pip install kagglehub
    python data/download_mnist.py
"""
import os
import shutil

import kagglehub

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    src = kagglehub.dataset_download("oddrationale/mnist-in-csv")
    for fname in ("mnist_train.csv", "mnist_test.csv"):
        path = os.path.join(src, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{fname} not found in {src}")
        out = os.path.join(DATA_DIR, fname)
        with open(path) as fin, open(out, "w") as fout:
            first = fin.readline()
            if first.split(",")[0].strip().isdigit():  # data_loader reads header=None
                fout.write(first)
            shutil.copyfileobj(fin, fout)
        print(f"saved data/{fname}")


if __name__ == "__main__":
    main()
