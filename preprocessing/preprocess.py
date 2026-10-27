"""Preprocessing steps for the GUI. Each takes and returns a uint8 grayscale image.

Noise-reduction filters come from Data_Preprocessing_methods.ipynb (Gaussian,
median and bilateral were compared there); Otsu turns the result into a binary
image with ink = 255, which segmentation needs. The GUI applies the ticked steps
in the order listed here.
"""
import cv2


def gaussian_blur(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def median_blur(gray):
    return cv2.medianBlur(gray, 5)


def bilateral_filter(gray):
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def otsu_threshold(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


METHODS = {
    "Gaussian blur": gaussian_blur,
    "Median blur": median_blur,
    "Bilateral filter": bilateral_filter,
    "Otsu threshold": otsu_threshold,
}
