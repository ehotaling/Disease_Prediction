import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained model
from data_utils import load_and_clean_data
from prediction_mapping import get_treatment

# ============================================================
# File: predict_cli.py
# Purpose: Provide a command-line interface (CLI) for making predictions.
#          The script loads a pre-trained model, accepts a symptom list
#          from the user, converts the input into the correct binary
#          feature vector, predicts the disease, and then looks up and
#          prints the recommended treatment.
# ============================================================

# -----------------------------
# Configuration and Paths
# -----------------------------
# Path to the pre-trained model.
# Ensure you run model_training.py first
model_path = "../models/rf_model.pkl"

# Path to the training data (to extract feature names and target mapping)
training_data_path = "../data/training_data.csv"

# -----------------------------
# Load the pre-trained model
# -----------------------------
try:
    model = joblib.load(model_path)
    print("Pre-trained model loaded successfully.")
except Exception as e:
    print("Error loading model. Please ensure the model file exists at:", model_path)
    raise e

# -----------------------------
# Load training data to get feature names and target mapping
# -----------------------------
df = load_and_clean_data(training_data_path, 'training')
# Extract the list of symptom feature columns (all columns except the target 'prognosis')
feature_cols = list(df.columns.drop('prognosis'))

# Factorize the target column to obtain a mapping of integer labels to disease names.
y_encoded, uniques = pd.factorize(df['prognosis'])
print("Target classes (label mapping):", uniques.tolist())


# -----------------------------
# Function to convert user symptoms into a binary input vector
# -----------------------------
def symptoms_to_vector(user_symptoms, feature_list):
    """
    Convert a comma-separated list of symptoms (input by the user) into
    a binary vector that aligns with the model's feature order.

    Parameters:
        user_symptoms (str): Comma-separated string of symptoms.
        feature_list (list): List of symptom features as used in training.

    Returns:
        np.array: A binary vector (1 if symptom is present, 0 otherwise).
    """
    # Convert user input to a set of lowercase symptom names for case-insensitive matching
    user_symptom_set = set(sym.strip().lower() for sym in user_symptoms.split(","))

    # Initialize a binary vector with zeros, length equal to the number of features
    input_vector = np.zeros(len(feature_list), dtype=int)

    # For each feature in the training data, set to 1 if it appears in the user input
    for i, symptom in enumerate(feature_list):
        if symptom.lower() in user_symptom_set:
            input_vector[i] = 1
    return input_vector


# -----------------------------
# Main prediction interface
# -----------------------------
def main():
    print("\nWelcome to the Disease Prediction CLI!")
    print("Enter the symptoms you are experiencing, separated by commas.")
    print("For example: fatigue, nausea, high_fever\n")

    # Get input from the user
    user_input = input("Enter symptoms (comma-separated): ")

    # Convert the user input into a binary vector matching the model's features
    input_vector = symptoms_to_vector(user_input, feature_cols)

    # Reshape the vector into a 2D array (1 sample, n features)
    input_vector = input_vector.reshape(1, -1)

    # ----- IMPORTANT FIX: Convert numpy array to DataFrame with proper feature names -----
    # This ensures the model sees the input in the same format it was trained on.
    input_df = pd.DataFrame(input_vector, columns=feature_cols)

    # Make a prediction using the pre-trained model
    predicted_label = model.predict(input_df)[0]

    # Convert the predicted integer label back to the disease name using the mapping from factorization
    predicted_disease = uniques[predicted_label]

    # Retrieve the recommended treatment using the prediction mapping function
    treatment_recommendation = get_treatment(predicted_disease)

    # Display the prediction and treatment information to the user
    print("\nPrediction Results:")
    print("-------------------")
    print("Predicted Disease: {}".format(predicted_disease))
    print("Recommended Treatment: {}".format(treatment_recommendation))


if __name__ == "__main__":
    main()
