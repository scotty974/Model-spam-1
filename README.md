Spam Detection Model

This project provides a spam detection model built using a K-Nearest Neighbors (KNN) classifier and TF-IDF vectorization. The model is trained on a spam dataset and can predict whether a given message is spam or not.


Project Overview

The project involves the following steps:

    Data Preprocessing: Load and clean the spam dataset.
    Feature Extraction: Convert text data into numerical features using TF-IDF vectorization.
    Model Training: Train a K-Nearest Neighbors (KNN) classifier using the processed data.
    Model Evaluation: Evaluate the model using cross-validation and validation curves.
    Model Optimization: Perform a grid search to find the best hyperparameters for the KNN classifier.
    Prediction: Use the trained model to predict whether new messages are spam or not.

Setup and Installation

    Clone the repository:

    bash

git clone https://github.com/yourusername/spam-detection-model.git
cd spam-detection-model

Usage
Training and Evaluating the Model

    Run the main script to train and save the model:

    bash

    python train.py

    This will train the model, perform grid search for hyperparameter tuning, and save the best model and vectorizer to disk.

Loading and Predicting

    Use the engine.py script to load the model and make predictions:

    bash

    python engine.py

    Modify the engine.py file to include the text you want to classify. The script will load the pre-trained model and vectorizer, and then evaluate whether the provided text is spam or not.

Files

    train.py: Script to train the KNN model and save it along with the TF-IDF vectorizer.
    engine.py: Script to load the model and vectorizer, and predict whether a text message is spam.
    spam.csv: The dataset used for training and evaluating the model.



