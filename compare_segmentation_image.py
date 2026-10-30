"""
Run the three segmentation methods on ONE image

    python compare_segmentation_image.py                    # uses IMAGE_PATH below


Needs only: numpy, opencv-python.

Prints the boxes each method finds and how long it took, and saves a picture with
the boxes drawn on. If you fill in TRUE_BOXES (hand-labelled actual boxes) it also
scores predicted vs actual with IoU.

Conventions: boxes are (x, y, w, h). Methods work on a binary image with ink = 255
on black (0). Light-background images are inverted automatically (see load_gray).
"""
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- config
IMAGE_PATH = os.path.join(ROOT, "multidigit_mnist.png")
OUT_PATH = os.path.join(ROOT, "results", "segmentation_single_image.png")
MIN_AREA = 20            # drop ink blobs smaller than this many pixels (speck removal)
MARKER_FRACTION = 0.3    # watershed: marker = distance >= this * blob's max distance
IOU_THRESHOLD = 0.5      # a predicted box matches a true box at IoU >= this

# Optional ground truth: one (x, y, w, h) per digit, in the ORIGINAL image's pixels,
# e.g. TRUE_BOXES = [(41, 38, 162, 251), (344, 37, 71, 251), (602, 120, 166, 232)]
# Leave as None to skip scoring.
TRUE_BOXES = None


# ---------------------------------------------------------------- loading + preprocessing
def load_gray(path):
    """Read as grayscale; if the background is light (dark ink on paper), invert so
    ink is bright on black."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return 255 - img if img.mean() > 127 else img


def to_binary(gray):
    """Blur -> Otsu threshold -> drop tiny specks."""
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
    return sorted((int(x), int(y), int(w), int(h)) for x, y, w, h, _ in stats[1:])


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
    at their narrowest point. Can over-split a single digit."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    n_blobs, blob_labels = cv2.connectedComponents(binary)
    sure_fg = np.zeros_like(binary)
    for b in range(1, n_blobs):
        mask = blob_labels == b
        core = mask & (dist >= marker_fraction * dist[mask].max())
        sure_fg[core if core.any() else mask] = 255
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers += 1                      # 1 = background, 2.. = digit cores
    markers[unknown == 255] = 0       # 0 = let the watershed decide
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


# ---------------------------------------------------------------- optional scoring
def _iou(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    inter = max(0, w) * max(0, h)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def score(pred, true):
    """Greedy one-to-one matching by IoU. Returns (tp, fp, fn, mean IoU of matches)."""
    pairs = sorted(((_iou(p, t), i, j) for i, p in enumerate(pred)
                    for j, t in enumerate(true)), reverse=True)
    used_p, used_t, ious = set(), set(), []
    for v, i, j in pairs:
        if v < IOU_THRESHOLD:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            ious.append(v)
    tp = len(ious)
    return tp, len(pred) - tp, len(true) - tp, (sum(ious) / tp if tp else 0.0)


# ---------------------------------------------------------------- picture
def draw(binary, pred, true, title):
    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in (true or []):                                  # green = actual
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 1)
    for x, y, w, h in pred:                                          # red = predicted
        cv2.rectangle(img, (x + 2, y + 2), (x + w - 2, y + h - 2), (0, 0, 255), 1)
    cv2.putText(img, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return img


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    gray = load_gray(path)
    binary = to_binary(gray)
    print(f"Image: {path}  size={gray.shape[1]}x{gray.shape[0]}  "
        f"ink pixels={int((binary > 0).sum())}")
    if TRUE_BOXES:
        print(f"Scoring against {len(TRUE_BOXES)} true boxes (match at IoU >= {IOU_THRESHOLD})")

    panels = []
    for name, method in METHODS.items():
        t0 = time.perf_counter()
        pred = method(binary)
        ms = (time.perf_counter() - t0) * 1000
        line = f"\n{name:22s} found {len(pred)} symbols in {ms:.1f} ms\n  boxes: {pred}"
        if TRUE_BOXES:
            tp, fp, fn, miou = score(pred, TRUE_BOXES)
            line += (f"\n  vs actual: count {'OK' if len(pred) == len(TRUE_BOXES) else 'WRONG'}"
                    f" | TP={tp} FP={fp} FN={fn} | mean IoU={miou:.3f}")
        print(line)
        panels.append(draw(binary, pred, TRUE_BOXES, f"{name} | {len(pred)} found"))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, np.vstack(panels))
    print(f"\nSaved picture to {OUT_PATH} (red = predicted"
        f"{', green = actual' if TRUE_BOXES else ''})")


if __name__ == "__main__":
    main()