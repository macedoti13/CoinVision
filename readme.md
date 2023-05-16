# Coin Classification and Conversion Project

## Overview

This project uses image processing techniques and machine learning to detect and classify coins in images, calculate their total value in Brazilian reais, and convert this value to other currencies. The project is implemented in Python, using OpenCV for image processing and scikit-learn for machine learning.

## Methodology

The project follows these main steps:

1. **Image Preprocessing:** Images are converted to grayscale and smoothed using Gaussian blur to reduce noise and detail.
2. **Coin Detection:* Coins are detected using the Hough Circle Transform implemented in OpenCV. This algorithm finds circular shapes in the image.
3. **Feature Extraction:** SIFT (Scale-Invariant Feature Transform) is used to detect and describe local features in the images. These features are essentially a set of keypoints and descriptors that capture the unique aspects of the image content around the detected keypoints, making them invariant to image scale, orientation, and affine distortion.
4. **Histogram of Visual Words:** We use a bag-of-visual-words model to represent each coin. We first build a codebook of visual words by clustering all descriptors using KMeans. Each cluster centroid represents a visual word. We then represent each coin by a histogram indicating the frequency of each visual word in the coin.
5. **Training a Classifier:** We use a Support Vector Machine (SVM) classifier to learn the mapping from the histogram of visual words to the coin class. The SVM model is trained on a labeled dataset and then saved for future use.
6. **Coin Classification:** We use the trained SVM model to classify the coins in new images.
7. **Currency Conversion:** We calculate the total value of the detected coins and convert this value to other currencies using the Exchange Rates API.

# How to Use

To use the project, simply run the `main.py` script. It will process the images in the images directory, detect and classify the coins, calculate the total value, convert the total value to other currencies, and log the results in the `log.txt` file.

The output is a log file containing the total value of coins detected in each image, both in Brazilian reais and in other currencies.

## How to Interpret the Results

The results in the `log.txt` file are organized as follows:

Image 1: 3.75 reais são equivalentes a:
   - 0.7 USD
   - 0.6 EUR
   - 0.5 GBP
   - 70.0 ARS
   - 4.5 CNY
   - 0.9 CAD

This means that in the first image, coins worth 3.75 reais were detected, and this value is equivalent to 0.7 US dollars, 0.6 euros, 0.5 British pounds, 70 Argentine pesos, 4.5 Chinese yuan, and 0.9 Canadian dollars.

## Technologies Used

- **Python:** The project is implemented in Python, a powerful and versatile programming language that is widely used in data science and machine learning.
- **OpenCV:** OpenCV (Open Source Computer Vision Library) is used for image processing tasks such as converting images to grayscale, applying Gaussian blur, and detecting circles.
- **scikit-learn:** scikit-learn is a popular machine learning library in Python. We use it to implement the SVM classifier and the KMeans clustering for creating the codebook of visual words.
- **SIFT:** SIFT (Scale-Invariant Feature Transform) is an algorithm in computer vision to detect and describe local features in images. It was patented by the University of British Columbia and since 2020 is free to use.
- **Exchange Rates API:** The Exchange Rates API is a free service for current and historical foreign exchange rates. We use it to convert the total value of coins from Brazilian reais to other currencies.

obs: This readme file was written by chatGPT