"""
Basic Gradio GUI for the Handwritten Number Recognition System.
Upload an image of a number/expression, or several single-digit images to be
joined into one number, then run preprocess -> segment -> classify -> evaluate.

Run from the project root:  python app.py
"""
import glob
import json
import os
import pickle

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from calculator import calculate
from data_loader import CLASS_NAMES, load_data
from mnist_format import to_mnist_format
try:  # teammates' modules; the app still runs if they're missing or empty
    from preprocessing import preprocess
    from segmentation import segment
except ImportError:
    preprocess = segment = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# The GUI lists whatever the teammates register in these dicts, so new
# features show up here without editing app.py:
#   preprocess.METHODS = {"Name": f(gray_image) -> image}   (applied in order)
#   segment.METHODS    = {"Name": f(binary_image) -> [{"bbox", "image"}, ...]}
def _default_preprocess(gray):
    """Fallback used when no preprocessing is registered: Otsu, ink = 255."""
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _merge_stacked(boxes):
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
                y_gap = max(y1, y2) - min(y1 + h1, y2 + h2)   # < 0 means they overlap vertically
                if x_overlap >= 0.5 * min(w1, w2) and -0.1 * min(h1, h2) <= y_gap <= max(w1, w2):
                    x, y = min(x1, x2), min(y1, y2)
                    boxes[i] = (x, y, max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y)
                    del boxes[j]
                    merged_any = True
                    break
            if merged_any:
                break
    return boxes


def _default_segment(binary):
    """Fallback used when no segmentation is registered: one box per contour,
    with vertically stacked pieces merged, left to right.
    """
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = [b for b in map(cv2.boundingRect, contours) if b[2] * b[3] >= 20]
    boxes = sorted(_merge_stacked(boxes))
    return [{"bbox": b, "image": binary[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]} for b in boxes]


PRE_METHODS = getattr(preprocess, "METHODS", {}) or {"Default (Otsu)": _default_preprocess}
SEG_METHODS = getattr(segment, "METHODS", {}) or {"Default (contours)": _default_segment}


def load_classifier(name):
    """Return predict(batch of flat 0-255 pixel rows) -> class indices for a
    saved model: sklearn {'model', 'scaler'} pickle (.pkl) or Keras CNN (.keras).
    """
    path = os.path.join(MODEL_DIR, name)
    if name.endswith(".keras"):
        from tensorflow import keras  # only imported when a CNN is picked
        cnn = keras.models.load_model(path)
        return lambda batch: np.argmax(
            cnn.predict(batch.reshape(-1, 28, 28, 1) / 255.0, verbose=0), axis=1)
    with open(path, "rb") as fh:
        saved = pickle.load(fh)
    scaler = saved["scaler"]
    return lambda batch: saved["model"].predict(scaler.transform(batch) if scaler else batch)


def digits_to_image(paths) -> np.ndarray:
    """Join single-digit images (sorted by filename) into one number image,
    with white gaps so segmentation can split them again.
    """
    tiles = []
    for p in sorted(paths):
        img = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (64, 64))
        tiles.append(cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255))
    return np.hstack(tiles)


def recognise(image, digit_files, model_name, pre_steps, seg_method):
    if digit_files:  # digit-folder mode takes priority when files are given
        image = digits_to_image(digit_files)
    elif image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # gradio gives RGB
    else:
        raise gr.Error("Upload an image or a set of digit images first.")

    predict = load_classifier(model_name)

    if not pre_steps or seg_method not in SEG_METHODS:
        raise gr.Error("Select at least one preprocessing step and a segmentation method.")
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    for name in PRE_METHODS:  # fixed order, not click order
        if name in pre_steps:
            binary = PRE_METHODS[name](binary)
    if binary.mean() > 127:  # models want white ink on black; ink is the minority, so flip
        binary = 255 - binary
    symbols = SEG_METHODS[seg_method](binary)
    if not symbols:
        return binary, [], "", "No symbols found."

    # same MNIST-style formatting as the training data; pixels stay 0-255
    # (the MLP's scaler was fitted on raw values, the CNN divides by 255 itself)
    crops = [to_mnist_format(s["image"]) for s in symbols]
    batch = np.array([c.reshape(-1) for c in crops], dtype=np.float32)
    labels = [CLASS_NAMES[int(p)] for p in predict(batch)]
    text = "".join(labels)

    try:
        result = str(calculate(text))
    except ValueError as e:  # e.g. a misread symbol made the expression invalid
        result = str(e)
    return binary, list(zip(crops, labels)), text, result


_mnist_test = None


def generate_number(n_digits):
    """Build a random multi-digit number image from the MNIST test set and
    show it in the image box. Swap this out for the team's generator when it
    is ready.
    """
    global _mnist_test
    if _mnist_test is None:  # read the CSV once, not on every click
        _mnist_test = load_data(normalize=False)[2]
    X = _mnist_test
    idx = np.random.choice(len(X), int(n_digits), replace=False)
    tiles = [cv2.copyMakeBorder(255 - cv2.resize(X[i], (64, 64)), 16, 16, 16, 16,
                                cv2.BORDER_CONSTANT, value=255) for i in idx]
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_GRAY2RGB)


RESULTS_DIR = os.path.join(MODEL_DIR, "results")


def load_metrics():
    """Read the {name}_metrics.json files the training scripts save: {model name: metrics}."""
    metrics = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_metrics.json"))):
        with open(path) as fh:
            m = json.load(fh)
        metrics[m["name"]] = m
    return metrics


def comparison_table():
    rows = []
    for name, m in load_metrics().items():
        cv = f"{m['cv_f1_mean']:.4f} +/- {m['cv_f1_std']:.4f}" if "cv_f1_mean" in m else "-"
        rows.append([name, f"{m['accuracy']:.4f}", f"{m['weighted_f1']:.4f}", cv])
    return pd.DataFrame(rows, columns=["Model", "Test accuracy", "Weighted F1", "5-fold CV F1"])


def show_model(name):
    """Confusion matrix, training history and per-class table for one model."""
    m = load_metrics().get(name)
    if m is None:
        return None, None, pd.DataFrame()

    def plot_path(kind):
        path = os.path.join(RESULTS_DIR, f"{name.lower()}_{kind}.png")
        return path if os.path.exists(path) else None

    per_class = pd.DataFrame(
        [[c, v["precision"], v["recall"], v["f1-score"], int(v["support"])]
         for c, v in m["per_class"].items() if c in CLASS_NAMES],
        columns=["Symbol", "Precision", "Recall", "F1", "Test images"]).round(3)
    return plot_path("confusion_matrix"), plot_path("training_history"), per_class


def refresh_comparison():
    names = list(load_metrics())
    return comparison_table(), gr.Dropdown(choices=names, value=names[0] if names else None)


models = [os.path.basename(p) for ext in ("*.pkl", "*.keras")
          for p in glob.glob(os.path.join(MODEL_DIR, ext))]

with gr.Blocks(title="Handwritten Number Recognition System") as demo:
    gr.Markdown("# Handwritten Number Recognition System")
    with gr.Tab("Recognise"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Image of a number / expression", type="numpy")
                files_in = gr.File(label="OR several single-digit images (joined in filename order)",
                                   file_count="multiple", file_types=["image"], type="filepath")
                with gr.Row():
                    n_digits = gr.Slider(1, 8, value=4, step=1, label="Digits")
                    gen_btn = gr.Button("Generate number from dataset")
                model_in = gr.Dropdown(models, value=models[0] if models else None, label="Model")
                pre_in = gr.CheckboxGroup(list(PRE_METHODS), value=list(PRE_METHODS),
                                          label="Preprocessing steps")
                seg_in = gr.Radio(list(SEG_METHODS), value=next(iter(SEG_METHODS), None),
                                  label="Segmentation method")
                run_btn = gr.Button("Recognise", variant="primary")
            with gr.Column():
                binary_out = gr.Image(label="Preprocessed (binary)")
                symbols_out = gr.Gallery(label="Segmented symbols + predictions", columns=8)
                text_out = gr.Textbox(label="Recognised text")
                result_out = gr.Textbox(label="Result")

    with gr.Tab("Model Comparison"):
        gr.Markdown("Saved by each `models/*_model.py` script when it is trained. "
                    "Retrain a model, then press Refresh.")
        table = gr.Dataframe(comparison_table(), interactive=False)
        refresh_btn = gr.Button("Refresh")
        names = list(load_metrics())
        pick = gr.Dropdown(names, value=names[0] if names else None, label="Show details for")
        with gr.Row():
            cm_img = gr.Image(label="Confusion matrix")
            hist_img = gr.Image(label="Training history")
        per_class_out = gr.Dataframe(label="Per-class results", interactive=False)

    gen_btn.click(generate_number, n_digits, image_in)
    run_btn.click(recognise, [image_in, files_in, model_in, pre_in, seg_in],
                  [binary_out, symbols_out, text_out, result_out])
    pick.change(show_model, pick, [cm_img, hist_img, per_class_out])
    refresh_btn.click(refresh_comparison, None, [table, pick])
    demo.load(show_model, pick, [cm_img, hist_img, per_class_out])

if __name__ == "__main__":
    demo.launch()
