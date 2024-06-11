# Handwritten Character Recognition

This project involves building a Convolutional Neural Network (CNN) model to recognize handwritten alphabets (A-Z) using the dataset from the [A_Z Handwritten Data.csv](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format).

## Introduction

This project aims to implement a deep learning model using Keras to classify handwritten alphabets. The model is trained on a dataset containing images of handwritten letters and can predict the correct letter when given an image.

## Dataset

The dataset used is `A_Z Handwritten Data.csv`, which contains pixel values of images of handwritten letters along with their corresponding labels.

- **Features**: Pixel values of the images (28x28).
- **Labels**: Alphabets (A-Z).

## Model Architecture

The Convolutional Neural Network (CNN) model consists of:

- Convolutional layers with ReLU activation
- Max Pooling layers
- Fully connected (Dense) layers with ReLU activation
- Output layer with softmax activation for classification

## Training

The model is trained using the Adam optimizer and categorical cross-entropy loss function. The learning rate is adjusted using the `ReduceLROnPlateau` callback, and early stopping is implemented to avoid overfitting.

## Results

The model achieves high accuracy on both training and validation sets. The performance is visualized using accuracy and loss plots.


