"""
Compare three image segmentation methods (predicted vs actual).

What it does
    1. Builds synthetic multi-digit numbers from the MNIST test CSV. Because we place
    every digit ourselves, we know the TRUE bounding box and label of each one.
    2. Makes them harder in steps: spaced -> tight -> touching -> overlapping -> noisy.
    3. Runs the SAME preprocessing on every image, then each segmentation method:
        a) connected components   b) vertical projection profile   c) watershed
    4. Scores predicted boxes against the true boxes (IoU matching), and optionally
    feeds the crops to a saved classifier to check the whole number end to end.
    5. Prints a results table, saves CSVs, and saves an example picture to results/segmentation/.

Conventions: images are uint8, ink = 255 on a black (0) background. Boxes are (x, y, w, h).
"""
import glob
import os
import pickle
import sys
import time
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
from data_loader import CLASS_NAMES, TEST_PATH   # noqa: E402
from mnist_format import INK_FLOOR, to_mnist_format   # noqa: E402

# ---------------------------------------------------------------- config
SEED = 42
N_NUMBERS = 100            # synthetic numbers per difficulty level
DIGIT_COUNTS = (2, 5)      # each number has between 2 and 5 digits (inclusive)
IOU_THRESHOLD = 0.5        # a predicted box "matches" a true box at IoU >= this
MIN_AREA = 20              # shared cleanup: drop ink blobs smaller than this many pixels
MARKER_FRACTION = 0.3      # watershed: marker = distance >= this * blob's max distance
MODEL_NAME = "random_forest_model.pkl"   # in models/; set to None to skip the end-to-end check
OUT_DIR = os.path.join(ROOT, "results", "segmentation")

# difficulty level -> (ink gap in px between neighbouring digits, noise sigma, speckle fraction)
# A negative gap means the digits' bounding boxes overlap by that many pixels.
LEVELS = {
    "spaced":      (6, 0, 0.0),
    "tight":       (2, 0, 0.0),
    "touching":    (-2, 0, 0.0),
    "overlapping": (-5, 0, 0.0),
    "noisy":       (3, 30, 0.01),
}


# ---------------------------------------------------------------- test-set generator
def build_number(digits, ink_gap, margin=10):
    """Paste 28x28 digits left to right so that the ink of neighbours is `ink_gap`
    pixels apart. Returns (grayscale canvas, true boxes)."""
    placements, prev_right = [], None
    for d in digits:
        ys, xs = np.nonzero(d > INK_FLOOR)
        left, right, top, bottom = xs.min(), xs.max(), ys.min(), ys.max()
        x_off = margin - left if prev_right is None else prev_right + 1 + ink_gap - left
        prev_right = x_off + right
        placements.append((x_off, left, right, top, bottom))

    shift = max(0, -min(p[0] for p in placements))      # keep every 28x28 tile inside the canvas
    width = max(p[0] for p in placements) + shift + 28 + margin
    canvas = np.zeros((28 + 2 * margin, width), np.uint8)
    boxes = []
    for d, (x_off, left, right, top, bottom) in zip(digits, placements):
        x_off += shift
        region = canvas[margin:margin + 28, x_off:x_off + 28]
        np.maximum(region, d, out=region)
        boxes.append((int(x_off + left), int(margin + top),
                    int(right - left + 1), int(bottom - top + 1)))
    return canvas, boxes


def add_noise(img, rng, sigma, speckle):
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    if speckle:
        m = rng.random(img.shape)
        out[m < speckle / 2] = 0
        out[m > 1 - speckle / 2] = 255
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- standardised stand-in preprocessing
def to_binary(gray):
    """Blur -> Otsu threshold -> drop tiny specks. Same for every method, so the
    comparison is about segmentation, not preprocessing."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean[labels == i] = 255
    return clean


# ---------------------------------------------------------------- the three methods
def seg_connected_components(binary):
    """Every 8-connected blob of ink is one symbol. Fails when digits touch (merge)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h, _ in stats[1:]]
    return sorted(boxes)


def seg_projection(binary):
    """Cut at columns with no ink at all. Fails when digits overlap horizontally."""
    has_ink = np.append((binary > 0).any(axis=0), False)
    boxes, start = [], None
    for x, on in enumerate(has_ink):
        if on and start is None:
            start = x
        elif not on and start is not None:
            rows = np.nonzero((binary[:, start:x] > 0).any(axis=1))[0]
            boxes.append((start, int(rows.min()), x - start, int(rows.max() - rows.min() + 1)))
            start = None
    return boxes


def seg_watershed(binary, marker_fraction=MARKER_FRACTION):
    """Distance transform -> blob cores as markers -> watershed splits touching digits
    at their narrowest point. Can over-split a single digit if the fraction is too high."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    n_blobs, blob_labels = cv2.connectedComponents(binary)
    sure_fg = np.zeros_like(binary)
    for b in range(1, n_blobs):
        mask = blob_labels == b
        core = mask & (dist >= marker_fraction * dist[mask].max())
        sure_fg[core if core.any() else mask] = 255      # thin strokes (a "1") may have no core
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers += 1                      # 1 = background, 2.. = digit cores
    markers[unknown == 255] = 0       # 0 = let the watershed decide

    # flood over the inverted distance map so basins are digit cores
    flood = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    markers = cv2.watershed(cv2.cvtColor(255 - flood, cv2.COLOR_GRAY2BGR), markers)

    boxes = []
    for label in range(2, n_markers + 1):
        ys, xs = np.nonzero((markers == label) & (binary > 0))
        if len(xs):
            boxes.append((int(xs.min()), int(ys.min()),
                        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
    return sorted(boxes)


METHODS = {
    "connected_components": seg_connected_components,
    "projection_profile": seg_projection,
    "watershed": seg_watershed,
}


def as_symbols(method):
    """Wrap a method as binary_image -> [{"bbox", "image"}, ...], the format app.py
    describes for segment.METHODS (check it matches your segmentation/segment.py)."""
    def wrapped(binary):
        return [{"bbox": b, "image": binary[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]}
                for b in method(binary)]
    return wrapped


# ---------------------------------------------------------------- scoring
def _inter(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _iou(a, b):
    i = _inter(a, b)
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def match_boxes(pred, true, thr=IOU_THRESHOLD):
    """Greedy one-to-one matching by IoU. Returns the IoUs of the matched pairs."""
    pairs = sorted(((_iou(p, t), i, j) for i, p in enumerate(pred)
                    for j, t in enumerate(true)), reverse=True)
    used_p, used_t, matched = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            matched.append(v)
    return matched


def count_merges_splits(pred, true):
    """merged: predicted boxes mostly covering >=2 true digits.
    split: true digits mostly covered by >=2 predicted boxes."""
    merged = sum(1 for p in pred
                 if sum(_inter(p, t) / (t[2] * t[3]) >= 0.5 for t in true) >= 2)
    split = sum(1 for t in true
                if sum(_inter(p, t) / (p[2] * p[3]) >= 0.5 for p in pred) >= 2)
    return merged, split


def load_classifier():
    if not MODEL_NAME:
        return None
    path = os.path.join(ROOT, "models", MODEL_NAME)
    if not os.path.exists(path):
        print(f"[note] {path} not found - skipping the end-to-end number check.")
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)          # {'model', 'scaler'}


def read_number(binary, boxes, saved):
    """Crop each box, format like MNIST, classify, join into a string (as app.py does)."""
    if not boxes:
        return ""
    crops = [to_mnist_format(binary[y:y + h, x:x + w]) for x, y, w, h in boxes]
    batch = np.array([c.reshape(-1) for c in crops], dtype=np.float32)
    if saved["scaler"] is not None:
        batch = saved["scaler"].transform(batch)
    return "".join(CLASS_NAMES[int(p)] for p in saved["model"].predict(batch))


# ---------------------------------------------------------------- picture of examples
def draw_example(binary, true_boxes, pred_boxes, title, scale=3):
    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for x, y, w, h in true_boxes:                                   # green = actual
        cv2.rectangle(img, (x * scale, y * scale), ((x + w) * scale, (y + h) * scale), (0, 200, 0), 1)
    for x, y, w, h in pred_boxes:                                   # red = predicted
        cv2.rectangle(img, (x * scale + 2, y * scale + 2),
                      ((x + w) * scale - 2, (y + h) * scale - 2), (0, 0, 255), 1)
    cv2.putText(img, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return img


def make_mosaic(cells):
    """cells: list of rows, each row a list of BGR images. Pads to one size and stacks."""
    h = max(c.shape[0] for row in cells for c in row)
    w = max(c.shape[1] for row in cells for c in row)
    rows = []
    for row in cells:
        padded = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, w - c.shape[1],
                                    cv2.BORDER_CONSTANT, value=(40, 40, 40)) for c in row]
        rows.append(np.hstack(padded))
    return np.vstack(rows)


# ---------------------------------------------------------------- main 
def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(TEST_PATH, header=None)
    labels = df.iloc[:, 0].to_numpy()
    images = df.iloc[:, 1:].to_numpy(np.uint8).reshape(-1, 28, 28)
    saved = load_classifier()

    rows, examples = [], {}
    for level, (gap, sigma, speckle) in LEVELS.items():
        for k in range(N_NUMBERS):
            n = int(rng.integers(DIGIT_COUNTS[0], DIGIT_COUNTS[1] + 1))
            pick = rng.choice(len(images), n, replace=False)
            canvas, true_boxes = build_number([images[i] for i in pick], gap)
            true_text = "".join(CLASS_NAMES[int(labels[i])] for i in pick)

            # how many digits are really joined (touching ink) before any noise
            n_blobs = cv2.connectedComponents((canvas > INK_FLOOR).astype(np.uint8))[0] - 1
            touching = n_blobs < n

            noisy = add_noise(canvas, rng, sigma, speckle) if (sigma or speckle) else canvas
            binary = to_binary(noisy)

            for name, method in METHODS.items():
                t0 = time.perf_counter()
                pred = method(binary)
                ms = (time.perf_counter() - t0) * 1000

                matched = match_boxes(pred, true_boxes)
                merged, split = count_merges_splits(pred, true_boxes)
                row = dict(level=level, method=name, n_true=n, n_pred=len(pred),
                        count_ok=len(pred) == n, tp=len(matched),
                        fp=len(pred) - len(matched), fn=n - len(matched),
                        iou_sum=sum(matched), merged=merged, split=split,
                        touching=touching, ms=ms)
                if saved is not None:
                    row["number_ok"] = read_number(binary, pred, saved) == true_text
                rows.append(row)
                if k == 0:
                    examples[(level, name)] = draw_example(
                        binary, true_boxes, pred, f"{level} | {name} | {len(pred)}/{n} digits")

    res = pd.DataFrame(rows)
    agg = {"n_true": "sum", "tp": "sum", "fp": "sum", "fn": "sum", "iou_sum": "sum",
        "merged": "sum", "split": "sum", "count_ok": "mean", "ms": "mean",
        "touching": "mean"}
    if saved is not None:
        agg["number_ok"] = "mean"

    def summarise(g):
        s = g.groupby(["level", "method"], sort=False).agg(agg)
        s["precision"] = s.tp / (s.tp + s.fp).clip(lower=1)
        s["recall"] = s.tp / (s.tp + s.fn).clip(lower=1)
        s["f1"] = 2 * s.precision * s.recall / (s.precision + s.recall).clip(lower=1e-9)
        s["mean_iou"] = s.iou_sum / s.tp.clip(lower=1)
        s = s.rename(columns={"count_ok": "count_acc", "touching": "touching_pct"})
        cols = ["count_acc", "precision", "recall", "f1", "mean_iou", "merged", "split",
                "ms", "touching_pct"] + (["number_ok"] if saved is not None else [])
        return s[cols]

    by_level = summarise(res)
    overall = res.assign(level="ALL")
    overall = summarise(overall)

    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, "per_image_results.csv"), index=False)
    by_level.to_csv(os.path.join(OUT_DIR, "summary_by_level.csv"))
    overall.to_csv(os.path.join(OUT_DIR, "summary_overall.csv"))
    mosaic = make_mosaic([[examples[(lv, m)] for m in METHODS] for lv in LEVELS])
    cv2.imwrite(os.path.join(OUT_DIR, "examples.png"), mosaic)

    pd.set_option("display.width", 200)
    print(f"\n{N_NUMBERS} numbers per level, {len(LEVELS)} levels, seed {SEED}, "
        f"match at IoU >= {IOU_THRESHOLD}\n")
    print(by_level.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nOverall:\n" + overall.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved CSVs and examples.png to {OUT_DIR}")
    print("In examples.png: green = actual boxes, red = predicted boxes.")


if __name__ == "__main__":
    main()