"""
Evaluation shared by mlp_model.py and cnn_model.py, so every model is scored
the same way and the results can be compared side by side in the report:

  - evaluate():       accuracy, weighted F1, classification report and a
                      confusion matrix (PNG; the numbers also go to a JSON file
                      that the GUI's Model Comparison tab reads)
  - plot_history():   loss / accuracy per epoch (saved as a PNG)
  - cross_validate(): 5-fold stratified CV, weighted F1

Plots are saved to models/results/ instead of shown, so the scripts can run
without a display.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _save(fig, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _update_metrics(name, **fields):
    """Merge fields into results/<name>_metrics.json (evaluate and cross_validate
    each add their part, so the file is updated rather than overwritten).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name.lower()}_metrics.json")
    metrics = {}
    if os.path.exists(path):
        with open(path) as fh:
            metrics = json.load(fh)
    metrics.update(name=name, **fields)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)


def evaluate(name, y_test, y_pred, class_names):
    """Print accuracy, weighted F1 and the per-class report, and save the confusion matrix."""
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"{name} - Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    _update_metrics(name, accuracy=acc, weighted_f1=f1,
                    per_class=classification_report(y_test, y_pred, target_names=class_names,
                                                    output_dict=True))

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} - confusion matrix")
    _save(fig, f"{name.lower()}_confusion_matrix.png")
    return acc, f1


def plot_history(name, loss=None, val_loss=None, acc=None, val_acc=None,
                 xlabel="epoch", val_label="validation"):
    """Plot whichever per-epoch curves the model provides (lists, one value per epoch)."""
    panels = [
        ("Loss", [("train", loss, "teal"), (val_label, val_loss, "orange")]),
        ("Accuracy", [("train", acc, "teal"), (val_label, val_acc, "orange")]),
    ]
    panels = [(t, c) for t, c in panels if any(v is not None for _, v, _ in c)]  # skip empty panels
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), squeeze=False)
    for ax, (title, curves) in zip(axes[0], panels):
        for label, values, colour in curves:
            if values is not None:
                ax.plot(values, color=colour, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend()
    fig.suptitle(f"{name} training history")
    _save(fig, f"{name.lower()}_training_history.png")


def cross_validate(name, fit_predict, X, y, subset=None, n_splits=5, seed=42):
    """5-fold stratified CV on (a stratified slice of) the training data.

    name is the model's name (as passed to evaluate). fit_predict(X_train,
    y_train, X_val) must train a fresh model and return
    its predictions for X_val. Prints and returns mean / std weighted F1.
    """
    if subset and subset < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=subset, stratify=y, random_state=seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        pred = fit_predict(X[train_idx], y[train_idx], X[val_idx])
        scores.append(f1_score(y[val_idx], pred, average="weighted"))
    scores = np.array(scores)
    print(f"\n{n_splits}-fold CV (weighted F1, n={len(X)}): "
          f"mean {scores.mean():.4f} (std {scores.std():.4f})")
    _update_metrics(name, cv_f1_mean=scores.mean(), cv_f1_std=scores.std(), cv_n=len(X))
    return scores.mean(), scores.std()
