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
import subprocess
import sys

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from calculator import calculate
from data_loader import CLASS_NAMES, load_data, load_operator_data
from mnist_format import to_mnist_format
from preprocessing import preprocess
from segmentation import segment

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# The GUI lists whatever is registered in preprocess.METHODS / segment.METHODS
PRE_METHODS = preprocess.METHODS
SEG_METHODS = segment.METHODS


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


def run_pipeline(image, predict, pre_steps, seg_method):
    """image is BGR or gray. Returns (binary, [(crop, label), ...], text, result)."""
    if not pre_steps or seg_method not in SEG_METHODS:
        raise gr.Error("Select at least one preprocessing step and a segmentation method.")
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = binary  # greyscale before any filtering: crops are cut from this, not from the 1-bit image
    for name in PRE_METHODS:  # fixed order, not click order
        if name in pre_steps:
            binary = PRE_METHODS[name](binary)
    if len(np.unique(binary)) > 2:  # no threshold step ticked: segmentation needs a binary image
        binary = PRE_METHODS["Otsu threshold"](binary)
    flipped = binary.mean() > 127  # models want white ink on black; ink is the minority, so flip
    if flipped:
        binary = 255 - binary
    ink = gray if flipped else 255 - gray  # greyscale, white ink on black
    symbols = SEG_METHODS[seg_method](binary)
    if not symbols:
        return binary, [], "", "No symbols found."

    # Training images are soft anti-aliased greyscale, and thresholding a thin stroke
    # to 1-bit then rescaling it changes its shape (a '(' turns into a '1'). So the
    # threshold only finds the symbols; each crop is cut from the greyscale, masked to
    # the symbol's own strokes so neighbours and background don't leak in, then mixed
    # 3:1 with the thresholded crop (pure greyscale lost a little accuracy on digits).
    # Pixels stay 0-255 (the MLP's scaler was fitted on raw values, the CNN divides by 255).
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crops = []
    for sym in symbols:
        x, y, w, h = sym["bbox"]
        mask = cv2.dilate(sym["image"], grow, iterations=2)
        soft = cv2.bitwise_and(ink[y:y + h, x:x + w], mask)
        crops.append(to_mnist_format((0.75 * soft + 0.25 * sym["image"]).astype(np.uint8)))
    batch = np.array([c.reshape(-1) for c in crops], dtype=np.float32)
    labels = [CLASS_NAMES[int(p)] for p in predict(batch)]
    text = "".join(labels)

    try:
        result = str(calculate(text))
    except ValueError as e:  # e.g. a misread symbol made the expression invalid
        result = str(e)
    return binary, list(zip(crops, labels)), text, result


def recognise(image, digit_files, model_name, pre_steps, seg_method):
    if digit_files:  # digit-folder mode takes priority when files are given
        image = digits_to_image(digit_files)
    elif image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # gradio gives RGB
    else:
        raise gr.Error("Upload an image or a set of digit images first.")
    return run_pipeline(image, load_classifier(model_name), pre_steps, seg_method)


_pools = None


def _symbol_pools():
    """{symbol: array of 28x28 white-on-black test images}. Digits come from the
    MNIST test CSV and operators from the operator test split, so generated
    images never use anything the models were trained on."""
    global _pools
    if _pools is None:  # read the data once, not on every click
        _, _, X, y = load_data(normalize=False)
        _, _, op_X, op_y = load_operator_data()
        X, y = np.concatenate([X, op_X]), np.concatenate([y, op_y])
        _pools = {name: X[y == i] for i, name in enumerate(CLASS_NAMES)}
    return _pools


def symbols_to_image(text, rng=np.random):
    """Draw one random test image per character of text, side by side with white gaps."""
    pools = _symbol_pools()
    tiles = []
    for ch in text:
        img = pools[ch][rng.randint(len(pools[ch]))]
        if not ch.isdigit():  # clean operator crops are hairlines; thicken like real pen strokes
            img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        tile = 255 - cv2.resize(img, (64, 64))
        tiles.append(cv2.copyMakeBorder(tile, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255))
    return np.hstack(tiles)


def random_number(n_digits, rng=np.random):
    first = str(rng.randint(1, 10)) if n_digits > 1 else str(rng.randint(10))
    return first + "".join(str(rng.randint(10)) for _ in range(n_digits - 1))


def random_expression(n_terms, rng=np.random):
    """e.g. '12+(7*3)-4': n_terms numbers of 1-2 digits joined by + - * /, with
    an optional bracketed group. Re-rolled if it divides by zero."""
    while True:
        terms = [random_number(rng.randint(1, 3), rng) for _ in range(n_terms)]
        if n_terms >= 3 and rng.rand() < 0.5:
            i = rng.randint(0, n_terms - 1)
            j = rng.randint(i + 2, n_terms + 1)
            terms[i], terms[j - 1] = "(" + terms[i], terms[j - 1] + ")"
        text = terms[0]
        for t in terms[1:]:
            text += rng.choice(list("+-*/")) + t
        try:
            calculate(text)
            return text
        except ValueError:
            continue


def generate_number(n_digits):
    """Build a random multi-digit number image from the MNIST test set and
    show it in the image box."""
    return cv2.cvtColor(symbols_to_image(random_number(int(n_digits))), cv2.COLOR_GRAY2RGB)


def generate_expression(n_terms):
    """Same, but a random arithmetic expression with operators and brackets."""
    return cv2.cvtColor(symbols_to_image(random_expression(int(n_terms))), cv2.COLOR_GRAY2RGB)


def batch_test(n_images, kind, size, model_name, pre_steps, seg_method, seed):
    """Generate n_images random numbers/expressions, run each through the full
    pipeline and compare with what was drawn. Returns (summary, per-image table)."""
    if not model_name:
        raise gr.Error("No trained model found in models/.")
    rng = np.random.RandomState(int(seed))
    predict = load_classifier(model_name)
    rows, n_text, n_seg, n_value, sym_ok, sym_total = [], 0, 0, 0, 0, 0
    for _ in range(int(n_images)):
        truth = random_expression(int(size), rng) if kind == "Expressions" else random_number(int(size), rng)
        _, _, text, result = run_pipeline(symbols_to_image(truth, rng), predict, pre_steps, seg_method)
        expected = str(calculate(truth))
        n_text += text == truth
        n_seg += len(text) == len(truth)
        n_value += result == expected
        sym_ok += sum(a == b for a, b in zip(text, truth))
        sym_total += len(truth)
        rows.append([truth, text, expected, result, "yes" if text == truth else "NO"])
    n = int(n_images)
    summary = pd.DataFrame([
        ["Exact text match", f"{n_text}/{n}", f"{n_text / n:.1%}"],
        ["Correct symbol count (segmentation)", f"{n_seg}/{n}", f"{n_seg / n:.1%}"],
        ["Correct final answer", f"{n_value}/{n}", f"{n_value / n:.1%}"],
        ["Symbols correct (by position)", f"{sym_ok}/{sym_total}", f"{sym_ok / sym_total:.1%}"],
    ], columns=["Measure", "Count", "Rate"])
    table = pd.DataFrame(rows, columns=["Drawn", "Read as", "Expected answer", "Got", "Match"])
    return summary, table


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


def list_models():
    """Saved model files, CNN (the chosen model) first so it is the default."""
    found = [os.path.basename(p) for ext in ("*.pkl", "*.keras")
             for p in glob.glob(os.path.join(MODEL_DIR, ext))]
    return sorted(found, key=lambda n: (not n.endswith(".keras"), n))


models = list_models()

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPTS = {"CNN": "cnn_model.py", "MLP": "mlp_model.py", "Random Forest": "random_forest_model.py"}
DATA_SCRIPTS = {"MNIST digits": "download_mnist.py", "Operator symbols (Kaggle)": "prepare_kaggle_operators.py"}


def run_script(path):
    """Run a python script from the project root, yielding its output so far."""
    log = f"$ python {os.path.relpath(path, ROOT)}\n"
    yield log
    proc = subprocess.Popen([sys.executable, "-u", path], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        log += line
        yield log[-20000:]
    code = proc.wait()
    yield log[-20000:] + (f"\nFinished OK." if code == 0 else f"\nFAILED (exit code {code}).")


def train_model(which):
    """Run a training script, then refresh every model list and the results."""
    for log in run_script(os.path.join(ROOT, "models", TRAIN_SCRIPTS[which])):
        yield log, gr.skip(), gr.skip(), gr.skip()
    names, new_models = list(load_metrics()), list_models()
    drop = gr.Dropdown(choices=new_models, value=new_models[0] if new_models else None)
    yield (log,
           gr.Dropdown(choices=names, value=which_metrics_name(which, names)), drop, drop)


def which_metrics_name(which, names):
    """Metrics name of the model just trained (falls back to the first)."""
    key = which.replace(" ", "").lower()
    return next((n for n in names if n.lower() == key), names[0] if names else None)


def download_data(which):
    yield from run_script(os.path.join(ROOT, "data", DATA_SCRIPTS[which]))

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
                    gen_btn = gr.Button("Generate number")
                    expr_terms = gr.Slider(2, 5, value=3, step=1, label="Numbers in expression")
                    expr_btn = gr.Button("Generate expression")
                model_in = gr.Dropdown(models, value=models[0] if models else None, label="Model")
                pre_in = gr.CheckboxGroup(list(PRE_METHODS), value=["Gaussian blur", "Otsu threshold"],
                                          label="Preprocessing steps")
                seg_in = gr.Radio(list(SEG_METHODS), value=next(iter(SEG_METHODS), None),
                                  label="Segmentation method")
                run_btn = gr.Button("Recognise", variant="primary")
            with gr.Column():
                binary_out = gr.Image(label="Preprocessed (binary)")
                symbols_out = gr.Gallery(label="Segmented symbols + predictions", columns=8)
                text_out = gr.Textbox(label="Recognised text")
                result_out = gr.Textbox(label="Result")

    with gr.Tab("Batch Test"):
        gr.Markdown("Generate many random numbers or expressions from the test images "
                    "(never the training data), run each through the full pipeline and "
                    "score the result against what was drawn.")
        with gr.Row():
            b_kind = gr.Radio(["Numbers", "Expressions"], value="Expressions", label="Generate")
            b_n = gr.Slider(5, 200, value=50, step=5, label="How many")
            b_size = gr.Slider(2, 6, value=3, step=1, label="Digits per number / numbers per expression")
            b_seed = gr.Number(value=0, precision=0, label="Seed (same seed = same test set)")
        with gr.Row():
            b_model = gr.Dropdown(models, value=models[0] if models else None, label="Model")
            b_pre = gr.CheckboxGroup(list(PRE_METHODS), value=["Gaussian blur", "Otsu threshold"],
                                     label="Preprocessing steps")
            b_seg = gr.Radio(list(SEG_METHODS), value=next(iter(SEG_METHODS), None),
                             label="Segmentation method")
        b_btn = gr.Button("Run batch test", variant="primary")
        b_summary = gr.Dataframe(label="Summary", interactive=False)
        b_table = gr.Dataframe(label="Every test", interactive=False)

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

    with gr.Tab("Train & Data"):
        gr.Markdown("Runs the project scripts for you and shows their output live. "
                    "Download the datasets first, then train. Training can take several minutes.")
        gr.Markdown("### 1. Download datasets")
        gr.Markdown("**Note:** the operator symbols dataset is a `.rar` archive. Install "
                    "[7-Zip](https://www.7-zip.org/) or unrar first, otherwise the extraction can "
                    "silently stall and you only get a partial download (some symbol folders missing).")
        with gr.Row():
            data_pick = gr.Dropdown(list(DATA_SCRIPTS), value=next(iter(DATA_SCRIPTS)), label="Dataset")
            data_btn = gr.Button("Download dataset")
        data_log = gr.Textbox(label="Download output", lines=8, max_lines=8, autoscroll=True)
        gr.Markdown("### 2. Train a model")
        with gr.Row():
            train_pick = gr.Dropdown(list(TRAIN_SCRIPTS), value="CNN", label="Model")
            train_btn = gr.Button("Train model", variant="primary")
        train_log = gr.Textbox(label="Training output", lines=12, max_lines=12, autoscroll=True)
        gr.Markdown("When training finishes, the results appear in the Model Comparison tab.")

    data_btn.click(download_data, data_pick, data_log)
    train_btn.click(train_model, train_pick, [train_log, pick, model_in, b_model]).then(
        show_model, pick, [cm_img, hist_img, per_class_out]).then(
        comparison_table, None, table)
    gen_btn.click(generate_number, n_digits, image_in)
    expr_btn.click(generate_expression, expr_terms, image_in)
    b_btn.click(batch_test, [b_n, b_kind, b_size, b_model, b_pre, b_seg, b_seed], [b_summary, b_table])
    run_btn.click(recognise, [image_in, files_in, model_in, pre_in, seg_in],
                  [binary_out, symbols_out, text_out, result_out])
    pick.change(show_model, pick, [cm_img, hist_img, per_class_out])
    refresh_btn.click(refresh_comparison, None, [table, pick])
    demo.load(show_model, pick, [cm_img, hist_img, per_class_out])

if __name__ == "__main__":
    demo.launch()
