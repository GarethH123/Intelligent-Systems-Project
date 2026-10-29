# Handwritten Number Recognition System

A small app that reads handwritten numbers and simple maths expressions from an
image. Give it a photo of `26-3*(17-4)` and it will read the symbols, show you
what it saw, and work out the answer.

Built for COS30018 Intelligent Systems (Option B).

## How it works

1. **Preprocess** – the image is turned to greyscale, blurred to remove noise, and
   thresholded to black and white.
2. **Segment** – the page is split into separate symbols by finding the connected
   blobs of ink, left to right. Symbols drawn in several strokes (like `÷`) are
   merged back together.
3. **Classify** – each symbol is shrunk to 28x28 (the MNIST format) and a trained
   model decides which of 16 classes it is: `0-9`, `+`, `-`, `*`, `/`, `(`, `)`.
4. **Calculate** – the recognised text is evaluated safely (no `eval`) and the
   answer is shown.

## Setup

You need Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Running it

```
python app.py
```

Then open the link it prints (usually http://127.0.0.1:7860).

### First time? Get the data and a model

The app has a **Train & Data** tab so you don't need to touch the command line:

1. **Download the datasets.** Click *Download dataset* once for MNIST digits and once
   for the operator symbols.
   > The operator dataset is a `.rar` file. Install [7-Zip](https://www.7-zip.org/)
   > (or unrar) first, otherwise the download can quietly stop halfway and some
   > symbols will be missing. You also need a Kaggle account set up for `kagglehub`.
2. **Train a model.** Pick CNN, MLP or Random Forest and click *Train model*.
   The output appears live, and the results show up in the *Model Comparison* tab
   when it finishes. The CNN is the one we recommend.

A trained CNN and MLP are already included in `models/`, so you can skip training
and just try the app.

## The tabs

| Tab | What it does |
|---|---|
| **Recognise** | Upload an image, or upload several single-digit images to be joined into one number. You can also generate a random number or expression from the test images. Choose the model, the preprocessing steps and the segmentation method, then press *Recognise*. |
| **Batch Test** | Generates lots of random numbers or expressions, runs each through the whole pipeline and scores it against what was drawn. Uses test images only, never training data. |
| **Model Comparison** | Accuracy, F1 and cross-validation for each model, plus confusion matrices, training curves and per-symbol results. |
| **Train & Data** | Download the datasets and train the models. |

## The three models

All three are trained on the same data and scored the same way.

| Model | Test accuracy | 5-fold CV F1 |
|---|---|---|
| CNN (default) | 97.8% | 92.2% |
| MLP | 96.9% | 92.5% |
| Random Forest | 97.5% | 94.9% |

The CNN has the best test accuracy, so it is the default. Cross-validation was run
with fewer epochs and a smaller sample to keep it quick, which is why its CV score is lower.

## Project layout

```
app.py             the Gradio app
calculator.py      safe evaluation of the recognised expression
data_loader.py     loads MNIST and the operator images
mnist_format.py    turns a cropped symbol into a 28x28 MNIST-style image
preprocessing/     blur and threshold methods
segmentation/      splitting an image into symbols
models/            training scripts, saved models, and results/ (metrics and plots)
data/              dataset download scripts (the data itself is not in git)
archive/           old notebooks from while we were building this - not needed to run the app
Images/            sample images
```

## Running things without the app

Each training script can also be run directly from the project root:

```
python models/cnn_model.py
python models/mlp_model.py
python models/random_forest_model.py
```

Saved models are written to `models/` and their results to `models/results/`.

## Notes

- The Random Forest model file is about 700 MB, so it is not committed to git.
  Train it from the app if you want to use it.
