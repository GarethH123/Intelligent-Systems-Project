"""
Random Forest for MNIST digits + handwritten operators - the tree-based model
in the comparison, trained on the same combined 16-class dataset (0-9,
+ - * / ( )) as the MLP and CNN and scored with the same evaluation
(models/evaluation.py).

  1. Define the forest (200 trees, sqrt features, out-of-bag score)
  2. Evaluate: accuracy, weighted F1, classification report, confusion matrix
  3. Cross-validate: 5-fold stratified, weighted F1
  4. Save the fitted model to models/random_forest_model.pkl

"""
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

# lets "import data_loader" work when this file is run straight from models.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_combined_data, CLASS_NAMES
from evaluation import cross_validate, evaluate, plot_history

# --- config ---
RANDOM_STATE = 42
RUN_CV = True         # set False to skip cross-validation and just train once
CV_SUBSET = 20000     # training rows to use for CV (None = all 60k, much slower)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_forest_model.pkl")


def build_rf():
    return RandomForestClassifier(
        n_estimators=200,          # number of trees
        criterion='gini',
        max_depth=None,            # grow trees fully; set e.g. 20 to shrink the model
        min_samples_leaf=1,
        max_features='sqrt',       # ~28 of 784 pixels considered per split
        bootstrap=True,
        oob_score=True,            # free validation estimate from out-of-bag samples
        n_jobs=-1,                 # use all CPU cores
        random_state=RANDOM_STATE,
    )


def accuracy_by_tree_count(clf, X, y):
    """Accuracy of the forest as trees are added one at a time. A forest has no
    epochs, so this is its version of a training curve: it shows where adding
    more trees stops helping.
    """
    X = X.astype(np.float32)
    summed_proba = np.zeros((len(X), len(clf.classes_)))
    accuracies = []
    for tree in clf.estimators_:
        summed_proba += tree.predict_proba(X)
        accuracies.append((clf.classes_[summed_proba.argmax(axis=1)] == y).mean())
    return accuracies


def main():
    # load_combined_data gives 28x28 images (digits + operators); the forest wants
    # each one as a flat 784-vector as it cannot directly accept multi-dimensional
    # input arrays. Raw 0-255 pixels (no scaling) because that is what app.py feeds
    # saved .pkl models, and trees don't care about the scale anyway.
    X_train, y_train, X_test, y_test = load_combined_data(normalize=False)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # 1. Train
    # Same class balancing as the MLP and CNN: operators have far fewer
    # examples than digits, so up-weight them.
    rf_clf = build_rf()
    rf_clf.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))

    # 2. Score on the test set
    y_pred = rf_clf.predict(X_test)
    evaluate('RandomForest', y_test, y_pred, CLASS_NAMES)
    plot_history('RandomForest', val_acc=accuracy_by_tree_count(rf_clf, X_test, y_test),
                 xlabel='trees', val_label='test')
    print(f'Out-of-bag score: {rf_clf.oob_score_:.4f}')

    # 3. Cross-validation on a stratified slice of the training data
    if RUN_CV:
        def fit_predict(X_tr, y_tr, X_val):
            fold_clf = build_rf()
            fold_clf.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))
            return fold_clf.predict(X_val)

        cross_validate('RandomForest', fit_predict, X_train, y_train, subset=CV_SUBSET, seed=RANDOM_STATE)

    # 4. Save the model (scaler is None - trees don't need one, but kept there to match the other models)
    with open(MODEL_PATH, 'wb') as fh:
        pickle.dump({'model': rf_clf, 'scaler': None}, fh)
    print(f'\nSaved fitted model to {MODEL_PATH}')


if __name__ == "__main__":
    main()
