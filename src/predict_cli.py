import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained model
from data_utils import load_and_clean_data
from prediction_mapping import get_treatment

# ============================================================
# File: predict_cli.py
# Purpose: Provide a command-line interface (CLI) for making predictions.
#          Loads a pre-trained model, accepts a symptom list from the user,
#          converts it into the correct binary feature vector, predicts the disease,
#          and then looks up and prints the recommended treatment.
# ============================================================

# -----------------------------
# Configuration and Paths
# -----------------------------
model_path = "../models/rf_model.pkl"
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
# Load training data to extract feature names and target mapping
# -----------------------------
df = load_and_clean_data(training_data_path, 'training')
feature_cols = list(df.columns.drop('prognosis'))

# Factorize the target column to obtain the mapping of integer labels to disease names.
y_encoded, uniques = pd.factorize(df['prognosis'])
print("Target classes (label mapping):", uniques.tolist())


# -----------------------------
# Function to convert user symptoms into a binary input vector
# -----------------------------
def symptoms_to_vector(user_symptoms, feature_list):
    """
    Convert a comma-separated list of symptoms (input by the user) into
    a binary vector that aligns with the model's feature order.

    This function normalizes the input by:
      - Stripping extra spaces,
      - Converting to lowercase,
      - Replacing hyphens '-' with underscores '_'.

    Parameters:
        user_symptoms (str): Comma-separated string of symptoms.
        feature_list (list): List of symptom features as used in training.

    Returns:
        np.array: A binary vector (1 if symptom is present, 0 otherwise).
    """
    # Normalize user input: strip, lower-case, and replace '-' with '_'
    user_symptom_set = set(sym.strip().lower().replace('-', '_') for sym in user_symptoms.split(","))

    # Initialize binary vector for all features
    input_vector = np.zeros(len(feature_list), dtype=int)

    for i, symptom in enumerate(feature_list):
        # Normalize the feature name similarly for comparison
        normalized_feature = symptom.lower().replace('-', '_')
        if normalized_feature in user_symptom_set:
            input_vector[i] = 1
    return input_vector


# -----------------------------
# Main prediction interface
# -----------------------------
def main():
    print("\nWelcome to the Disease Prediction CLI!")
    print("Enter the symptoms you are experiencing, separated by commas.")
    print("For example: fatigue, nausea, high_fever\n")

    # Get user input
    user_input = input("Enter symptoms (comma-separated): ")

    # Convert input to binary vector and then to a DataFrame to preserve feature names.
    input_vector = symptoms_to_vector(user_input, feature_cols).reshape(1, -1)
    input_df = pd.DataFrame(input_vector, columns=feature_cols)

    # Predict the disease using the pre-trained model.
    predicted_label = model.predict(input_df)[0]

    # Convert predicted label back to disease name.
    predicted_disease = uniques[predicted_label]

    # Retrieve recommended treatment using fuzzy matching.
    treatment_recommendation = get_treatment(predicted_disease)

    # Display prediction and treatment.
    print("\nPrediction Results:")
    print("-------------------")
    print("Predicted Disease: {}".format(predicted_disease))
    print("Recommended Treatment: {}".format(treatment_recommendation))


if __name__ == "__main__":
    main()
