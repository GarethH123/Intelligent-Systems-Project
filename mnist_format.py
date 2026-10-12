"""
Normalises one symbol image to the same format MNIST digits are stored in, so
the operator training data looks like the MNIST digits the models are trained
on. Only used by data_loader for the training data; preprocessing/ and
segmentation/ stay input-image only:

  - tight crop around the ink
  - longest side scaled to 20 px, aspect ratio kept (anti-aliased, so grey edges)
  - pasted into a black 28x28 frame
  - brightness stretched so the strongest pixel is 255 (every MNIST image peaks at ~255)
  - shifted so the ink's centre of mass sits at (14, 14)
  - no thresholding afterwards -- MNIST is grayscale, not binary

I measured these on the MNIST test CSV: the longest side is always 20 px and
the mean centre of mass is (14.0, 14.0), and every image peaks at 254-255.

    from mnist_format import to_mnist_format
    img28 = to_mnist_format(symbol)   # symbol: 2D uint8, ink bright on black
"""

import cv2
import numpy as np

INK_FLOOR = 32   # pixels at or below this count as background when finding the crop box


def to_mnist_format(img: np.ndarray, size: int = 28, box: int = 20) -> np.ndarray:
    """img is a 2D uint8 image with the ink bright (high) on a dark background.
    Returns a size x size uint8 image in MNIST style. A blank image comes
    back as all zeros.
    """
    canvas = np.zeros((size, size), dtype=np.uint8)

    ys, xs = np.nonzero(img > INK_FLOOR)
    if len(xs) == 0:
        return canvas
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = crop.shape
    scale = box / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    # INTER_AREA gives the soft anti-aliased edges when shrinking; it doesn't
    # work well for enlarging (e.g. a tiny dot), so use linear there
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    # shrinking a thin stroke (a '-', a '(') averages it with the background and
    # leaves it faint, so stretch back up to full brightness like MNIST
    resized = np.clip(resized.astype(np.float32) * (255.0 / resized.max()), 0, 255).astype(np.uint8)

    x0, y0 = (size - new_w) // 2, (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    # move the centre of mass to the middle of the frame. Whole-pixel shift so
    # it doesn't blur the strokes again, and warpAffine (not np.roll) so
    # nothing wraps around the edge
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        dx = round(size / 2 - m["m10"] / m["m00"])
        dy = round(size / 2 - m["m01"] / m["m00"])
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, shift, (size, size))
    return canvas
