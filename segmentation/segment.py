"""Segmentation methods for the GUI: f(binary_image, ink = 255) -> [{"bbox", "image"}, ...]
sorted left to right."""
import cv2


def merge_stacked(boxes):
    """Merge boxes that sit directly above/below each other (same x-range, small
    vertical gap) into one, so symbols drawn in several strokes stay whole:
    the two dots and bar of a division sign, the two bars of an '=', etc.
    """
    boxes = list(boxes)
    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[j]
                x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
                y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)   # < 0 is the vertical gap between them
                stacked = x_overlap >= 0.3 * min(w1, w2) and -max(w1, w2) <= y_overlap <= 0.1 * min(h1, h2)
                # a stroke that mostly sits inside another piece's box (e.g. the far half of
                # a broken bar of a division sign) belongs to it
                inside = (max(x_overlap, 0) * max(y_overlap, 0)) >= 0.3 * min(w1 * h1, w2 * h2)
                if stacked or inside:
                    x, y = min(x1, x2), min(y1, y2)
                    boxes[i] = (x, y, max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y)
                    del boxes[j]
                    merged_any = True
                    break
            if merged_any:
                break
    return boxes


def connected_components_segment(binary):
    """One box per 8-connected blob of ink, with vertically stacked pieces merged,
    left to right."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h, _ in stats[1:]   # row 0 = background
             if w * h >= 20]
    boxes = sorted(merge_stacked(boxes))
    return [{"bbox": b, "image": binary[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]} for b in boxes]


METHODS = {"Connected components (merge stacked)": connected_components_segment}

'''
def contour_segment(binary):
    """One box per contour, with vertically stacked pieces merged, left to right."""
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = [b for b in map(cv2.boundingRect, contours) if b[2] * b[3] >= 20]
    boxes = sorted(merge_stacked(boxes))
    return [{"bbox": b, "image": binary[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]} for b in boxes]
'''
