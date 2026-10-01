"""
Loads MNIST digits from the CSVs plus the handwritten operator symbols
from data/operators/ (built by data/prepare_kaggle_operators.py), and
can combine the two into one dataset for the classifier.
"""
import os
import cv2
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "mnist_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "mnist_test.csv")
OPERATOR_DIR = os.path.join(DATA_DIR, "operators")

# maps label 10+ to (folder under data/operators/, the symbol itself)
OPERATOR_CLASSES = [
    ("plus", "+"),
    ("minus", "-"),
    ("times", "*"),
    ("div", "/"),
    ("lparen", "("),
    ("rparen", ")"),
]
CLASS_NAMES = [str(d) for d in range(10)] + [sym for _, sym in OPERATOR_CLASSES]


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


def _image_to_mnist_style(img: np.ndarray, size: int = 28) -> np.ndarray:
    """Resize an operator crop to match MNIST's look (28x28, ink=255 on black)."""
    # kaggle crops are dark ink on white, opposite of MNIST, so flip if needed
    if img.mean() > 127:
        img = 255 - img
    # resize before thresholding, not after -- otherwise thin strokes like a
    # '-' get averaged away during the downscale and just disappear
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def load_operator_data(test_fraction: float = 0.2, seed: int = 42,
                        operator_dir: str = OPERATOR_DIR, min_ink_px: int = 15):
    """Loads operator images from data/operators/<folder>/*.png and splits
    each class into train/test. Labels are 10-15, in OPERATOR_CLASSES order.

    min_ink_px throws out crops that end up basically blank after resizing --
    we were seeing bad '+' recall traced back to a bunch of these near-empty
    images in training.
    """
    rng = np.random.default_rng(seed)
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for label_offset, (folder, symbol) in enumerate(OPERATOR_CLASSES):
        label = 10 + label_offset
        folder_path = os.path.join(operator_dir, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Missing {folder_path} (symbol {symbol!r}). Run "
                "data/prepare_kaggle_operators.py to populate data/operators/ "
                "before loading the combined dataset."
            )

        fnames = sorted(os.listdir(folder_path))
        images = []
        for fname in fnames:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            normalized = _image_to_mnist_style(img)
            if (normalized > 127).sum() < min_ink_px:
                continue
            images.append(normalized)

        idx = rng.permutation(len(images))
        n_test = max(1, int(len(images) * test_fraction))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        train_images.extend(images[i] for i in train_idx)
        train_labels.extend([label] * len(train_idx))
        test_images.extend(images[i] for i in test_idx)
        test_labels.extend([label] * len(test_idx))

    X_train = np.stack(train_images).astype(np.uint8)
    y_train = np.array(train_labels, dtype=np.int64)
    X_test = np.stack(test_images).astype(np.uint8)
    y_test = np.array(test_labels, dtype=np.int64)
    return X_train, y_train, X_test, y_test


def load_combined_data(normalize: bool = True, seed: int = 42):
    """Combines MNIST digits (0-9) with the operator symbols (10-15) into
    one dataset, so the classifier can recognise anything segment_symbols()
    hands it from a full expression.
    """
    digit_X_train, digit_y_train, digit_X_test, digit_y_test = load_data(normalize=False)
    op_X_train, op_y_train, op_X_test, op_y_test = load_operator_data(seed=seed)

    # shuffle so operators aren't all bunched at the end of the array
    rng = np.random.default_rng(seed)

    X_train = np.concatenate([digit_X_train, op_X_train])
    y_train = np.concatenate([digit_y_train, op_y_train])
    train_shuffle = rng.permutation(len(X_train))
    X_train, y_train = X_train[train_shuffle], y_train[train_shuffle]

    X_test = np.concatenate([digit_X_test, op_X_test])
    y_test = np.concatenate([digit_y_test, op_y_test])
    test_shuffle = rng.permutation(len(X_test))
    X_test, y_test = X_test[test_shuffle], y_test[test_shuffle]

    # normalize after combining so both sources scale the same way
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")