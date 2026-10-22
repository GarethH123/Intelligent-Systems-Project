"""
CNN for MNIST digits + handwritten operators - the convolutional model in the
comparison, trained on the same combined 16-class dataset (0-9, + - * / ( ))
as the MLP and scored with the same evaluation (models/evaluation.py).

  1. Define the CNN (3 x conv+maxpool -> dense 256 -> softmax), from CNNFinished.ipynb
  2. Train with a validation split and early stopping (best weights restored)
  3. Evaluate: accuracy, weighted F1, classification report, confusion matrix
  4. Cross-validate: 5-fold stratified, weighted F1
  5. Save the fitted model to models/cnn_model.keras and the training curves

"""
import os
import sys

import numpy as np
from tensorflow import keras
from sklearn.utils.class_weight import compute_sample_weight

# lets "import data_loader" work when this file is run straight from models/.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_combined_data, CLASS_NAMES
from evaluation import cross_validate, evaluate, plot_history

layers = keras.layers

# --- config ---
RANDOM_STATE = 42
EPOCHS = 15           # upper limit, early stopping usually ends sooner
BATCH_SIZE = 128
PATIENCE = 3          # stop if val_loss hasn't improved for this many epochs
RUN_CV = True         # set False to skip cross-validation and just train once
CV_SUBSET = 20000     # training rows to use for CV (None = all, much slower)
CV_EPOCHS = 5         # epochs per fold, kept low because every fold trains a CNN
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model.keras")


def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # 16 filters, 3x3 kernel, stride 1, ReLU, then max pool halves the size
        layers.Conv2D(16, (3, 3), 1, activation='relu'),
        layers.MaxPooling2D(),
        # double the filters to pick up more complex shapes
        layers.Conv2D(32, (3, 3), 1, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, (3, 3), 1, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(CLASS_NAMES), activation='softmax'),   # one output per class
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    keras.utils.set_random_seed(RANDOM_STATE)

    # load_combined_data gives scaled 28x28 images (digits + operators); the CNN
    # wants a channel axis on the end, (N, 28, 28, 1)
    X_train, y_train, X_test, y_test = load_combined_data(normalize=True)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # 1. Train
    # Same class balancing as the MLP: operator classes have ~25x fewer
    # examples than digits, so up-weight them in the loss.
    sample_weight = compute_sample_weight('balanced', y_train)
    cnn = build_cnn()
    cnn.summary()
    # validation comes from the training data (not the test set), same as the
    # MLP's validation_fraction=0.2
    hist = cnn.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True)],
        verbose=2,
    )
    print(f'Training stopped after {len(hist.history["loss"])} epochs')

    # 2. Score on the test set
    y_pred = np.argmax(cnn.predict(X_test, verbose=0), axis=1)
    evaluate('CNN', y_test, y_pred, CLASS_NAMES)
    h = hist.history
    plot_history('CNN', h['loss'], h['val_loss'], h['accuracy'], h['val_accuracy'])

    # 3. Cross-validation on a stratified slice of the training data
    if RUN_CV:
        def fit_predict(X_tr, y_tr, X_val):
            fold = build_cnn()
            fold.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr),
                     epochs=CV_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            return np.argmax(fold.predict(X_val, verbose=0), axis=1)

        cross_validate('CNN', fit_predict, X_train, y_train, subset=CV_SUBSET, seed=RANDOM_STATE)

    # 4. Save the model (Keras format, not pickle; app.py loads it from there)
    cnn.save(MODEL_PATH)
    print(f'\nSaved fitted model to {MODEL_PATH}')


if __name__ == "__main__":
    main()
