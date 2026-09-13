"""
MLP baseline for MNIST - the "traditional neural network" in the model comparison.

  1. Define the MLP (64 -> 32 -> 16, ReLU, adam, early stopping)
  2. Evaluate: accuracy, weighted F1, classification report
  3. Cross-validate: 5-fold stratified, weighted F1
  4. Save the fitted model (and scaler) to models/mlp_model.pkl

"""
import os
import pickle
import sys

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# loads "import data_loader" work when this file is run straight from models/.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

# --- config ---
RANDOM_STATE = 42
USE_SCALER = True     # scale pixels with StandardScaler before training
RUN_CV = True         # set False to skip cross-validation and just train once
CV_SUBSET = 20000     # training rows to use for CV (None = all 60k, much slower)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlp_model.pkl")


def build_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),   # three hidden layers, getting smaller
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,               # stop if no improvement for 10 rounds
        random_state=RANDOM_STATE,
    )


def main():
    # load_data gives us 28x28 images; the MLP wants each one as a flat 784-vector.
    # If we're scaling below, load the raw pixels; if not, let load_data do the /255.
    X_train, y_train, X_test, y_test = load_data(normalize=not USE_SCALER)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    scaler = None
    if USE_SCALER:
        # Fit on train only, then apply the same transform to test.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 1. Train
    mlp_clf = build_mlp()
    mlp_clf.fit(X_train, y_train)

    # 2. Score on the test set
    y_pred = mlp_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    digit_names = [str(i) for i in range(10)]
    print(f'MLP Classifier - Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}')
    print(f'Training stopped at iteration: {mlp_clf.n_iter_}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=digit_names))

    # 3. Cross-validation on a stratified slice of the training data
    if RUN_CV:
        X_cv, y_cv = X_train, y_train
        if CV_SUBSET and CV_SUBSET < len(X_train):
            X_cv, _, y_cv, _ = train_test_split(
                X_train, y_train,
                train_size=CV_SUBSET,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        # Run folds one at a time. Parallel mode prints a wall of harmless
        cv_scores = cross_val_score(
            build_mlp(), X_cv, y_cv, cv=skf, scoring='f1_weighted', n_jobs=1
        )
        print(
            f'\n5-fold CV (weighted F1, n={len(X_cv)}): '
            f'mean {cv_scores.mean():.4f} (std {cv_scores.std():.4f})'
        )

    # 4. Save the model and scaler together, since predictions later need both
    with open(MODEL_PATH, 'wb') as fh:
        pickle.dump({'model': mlp_clf, 'scaler': scaler}, fh)
    print(f'\nSaved fitted model to {MODEL_PATH}')


if __name__ == "__main__":
    main()
