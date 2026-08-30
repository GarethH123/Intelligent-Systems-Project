"""
Loads MNIST from data/mnist_train.csv and data/mnist_test.csv.

Expected layout (standard MNIST CSV): column 0 = label (0-9),
columns 1-784 = pixel values (0-255), row-major.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "mnist_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "mnist_test.csv")


def _load_csv(path: str, normalize: bool):
    """Read one MNIST CSV file, split into (X, y), reshape pixels to 28x28."""
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32 if normalize else np.uint8)
    X = X.reshape(-1, 28, 28)
    if normalize:
        X /= 255.0
    return X, y


def load_data(normalize: bool = True):
    """Load train and test CSVs, return (X_train, y_train, X_test, y_test)."""
    X_train, y_train = _load_csv(TRAIN_PATH, normalize)
    X_test, y_test = _load_csv(TEST_PATH, normalize)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")