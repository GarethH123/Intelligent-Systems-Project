
import os
import pickle
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# imports data_loader.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

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


def main():
    # load_data gives 28x28 images; the forest wants each one as a flat 784-vector as it cannot directly accept multi-dimensional input arrays.
    X_train, y_train, X_test, y_test = load_data(normalize=True)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # 1. Train
    rf_clf = build_rf()
    rf_clf.fit(X_train, y_train)

    # 2. Score on the test set
    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    digit_names = [str(i) for i in range(10)]
    print(f'Random Forest - Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}')
    print(f'Out-of-bag score: {rf_clf.oob_score_:.4f}')
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
        # Folds run one at a time; stratifiess folds 5 times over based of n_split.
        cv_scores = cross_val_score(
            build_rf(), X_cv, y_cv, cv=skf, scoring='f1_weighted', n_jobs=1
        )
        print(
            f'\n5-fold CV (weighted F1, n={len(X_cv)}): '
            f'mean {cv_scores.mean():.4f} (std {cv_scores.std():.4f})'
        )

    # 4. Save the model (scaler is None - trees don't need one, but kept there just in case to be consistent with other models potentially)
    with open(MODEL_PATH, 'wb') as fh:
        pickle.dump({'model': rf_clf, 'scaler': None}, fh)
    print(f'\nSaved fitted model to {MODEL_PATH}')


if __name__ == "__main__":
    main()