# Star Classification Neural Network

This project involves a neural network model designed to classify astronomical objects into three classes: GALAXY, STAR, and QSO (Quasi-Stellar Objects). The model is built and trained using PyTorch.

## Project Overview

The goal of this project is to accurately classify astronomical objects based on their features such as 'u', 'g', 'r', 'i', 'z', and 'redshift'. The dataset used is `star_classification.csv`, which contains labeled examples of each class.

## Model Architecture

The model is a feedforward neural network with the following architecture:
- Input Layer: 6 neurons (one for each feature)
- First Hidden Layer: 12 neurons with ReLU activation
- Second Hidden Layer: 12 neurons with ReLU activation
- Output Layer: 3 neurons (one for each class)

## Dependencies

- PyTorch
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- joblib
- tqdm

## Dataset

The dataset `star_classification.csv` contains the following columns:
- 'u', 'g', 'r', 'i', 'z': Photometric magnitudes in different bands
- 'redshift': Redshift of the object
- 'class': The class label (GALAXY, STAR, QSO)

## Setup and Installation

Ensure that you have Python 3.x installed along with the necessary libraries. You can install the required packages using:

```bash
pip install torch pandas numpy scikit-learn matplotlib tqdm
```

## Usage

To train the model and get training data, run the main script:

```bash
python train_model.py
```

This will train the model and save the trained model as `model.pth`. The script will also generate a plot of the loss curve during training and validation, saved as `loss_curve.png` and a confusion matrix as `confusion_matrix.png`.

To manually test the model with new stellar object data, run the following script:

```bash
python predict.py
```

A user may then input star photometric data and load the saved `model.pth` and `scaler.save` to manually use it to classify a stellar object.

The prompt will be:
Enter the values for u, g, r, i, z, redshift

User input should be in the following format: 25.26307 22.66389 20.60976 20.25615 19.54544 1.424659


## Results

The model achieves a high accuracy of 97.09% on the test set, with both training and validation losses showing a consistent decrease over the training epochs.


## Credits

- Dataset from Sloan Digital Sky Survey
- https://www.sdss4.org/dr17/
- Pytorch nn documentation
- https://www.sdss4.org/dr17/
- Sentdex Youtube - Neural Networks from Scratch
- https://www.youtube.com/@sentdex
