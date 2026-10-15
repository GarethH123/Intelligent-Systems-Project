"""
Basic Gradio GUI for the Handwritten Number Recognition System.
Upload an image of a number/expression, or several single-digit images to be
joined into one number, then run preprocess -> segment -> classify -> evaluate.

Run from the project root:  python app.py
"""
import glob
import os
import pickle
import re

import cv2
import gradio as gr
import numpy as np

from data_loader import CLASS_NAMES, load_data
from mnist_format import to_mnist_format
from preprocessing import preprocess
from segmentation import segment

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# The GUI lists whatever the teammates register in these dicts, so new
# features show up here without editing app.py:
#   preprocess.METHODS = {"Name": f(gray_image) -> image}   (applied in order)
#   segment.METHODS    = {"Name": f(binary_image) -> [{"bbox", "image"}, ...]}
PRE_METHODS = getattr(preprocess, "METHODS", {})
SEG_METHODS = getattr(segment, "METHODS", {})


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

    with open(os.path.join(MODEL_DIR, model_name), "rb") as fh:
        saved = pickle.load(fh)  # {'model', 'scaler'}

    if not pre_steps or seg_method not in SEG_METHODS:
        raise gr.Error("Select at least one preprocessing step and a segmentation method.")
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    for name in PRE_METHODS:  # fixed order, not click order
        if name in pre_steps:
            binary = PRE_METHODS[name](binary)
    symbols = SEG_METHODS[seg_method](binary)
    if not symbols:
        return binary, [], "", "No symbols found."

    # same MNIST-style formatting as the training data; pixels stay 0-255
    # because the scaler was fitted on raw values
    crops = [to_mnist_format(s["image"]) for s in symbols]
    batch = np.array([c.reshape(-1) for c in crops], dtype=np.float32)
    if saved["scaler"] is not None:
        batch = saved["scaler"].transform(batch)
    labels = [CLASS_NAMES[int(p)] for p in saved["model"].predict(batch)]
    text = "".join(labels)

    try:  # only digits and + - * / ( ) can reach eval
        result = eval(text, {"__builtins__": {}}) if re.fullmatch(r"[0-9+\-*/()]+", text) else None
    except Exception:
        result = None
    return binary, list(zip(crops, labels)), text, "Not a valid expression" if result is None else str(result)


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


models = [os.path.basename(p) for p in glob.glob(os.path.join(MODEL_DIR, "*.pkl"))]

with gr.Blocks(title="Handwritten Number Recognition System") as demo:
    gr.Markdown("# Handwritten Number Recognition System")
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Image of a number / expression", type="numpy")
            files_in = gr.File(label="OR several single-digit images (joined in filename order)",
                               file_count="multiple", file_types=["image"], type="filepath")
            with gr.Row():
                n_digits = gr.Slider(2, 8, value=4, step=1, label="Digits")
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

    gen_btn.click(generate_number, n_digits, image_in)
    run_btn.click(recognise, [image_in, files_in, model_in, pre_in, seg_in],
                  [binary_out, symbols_out, text_out, result_out])

if __name__ == "__main__":
    demo.launch()
