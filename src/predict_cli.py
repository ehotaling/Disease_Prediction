import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained model
from torch.utils.data.datapipes.gen_pyi import split_outside_bracket

from data_utils import load_and_clean_data
from prediction_mapping import get_treatment
import os
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# File: predict_cli.py
# Purpose: Provide a command-line interface (CLI) for making predictions.
#          Utilizes GPT-4o mini to interpret user-input symptoms,
#          loads a pre-trained model, predicts the disease,
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
    print(f"Error loading model. Please ensure the model file exists at: {model_path}")
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
# Initialize OpenAI API
# -----------------------------
load_dotenv()  # This loads environment variables from the .env file
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def interpret_symptoms_with_gpt(user_input, feature_list):
    """
    Uses GPT-4o mini to interpret user-described symptoms and map them to standardized symptom features.

    Parameters:
        user_input (str): Free-form text describing symptoms.
        feature_list (list): List of standardized symptom features.

    Returns:
        list: Matched symptom features.
    """

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a medical professional that specializes symptom recognition",
            input=f"A patient describes their symptoms as: \"{user_input}\"\n\n"
                  f"From the following list of standardized medical symptoms:\n"
                  f"{', '.join(feature_list)}\n\n"
                  "Which symptoms from the list best match the patient's description? "
                  "Return a comma-separated list of the matching symptom keywords."
        )

        matched_symptoms = response.output_text.split(', ')
        return [symptom for symptom in matched_symptoms if symptom in feature_list]
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return []

def symptoms_to_vector(matched_symptoms, feature_list):
    """
    Convert a list of matched symptoms into a binary vector that aligns with the model's feature order.

    Parameters:
        matched_symptoms (list): List of symptoms matched from user input.
        feature_list (list): List of symptom features as used in training.

    Returns:
        np.array: A binary vector (1 if symptom is present, 0 otherwise).
    """
    input_vector = np.zeros(len(feature_list), dtype=int)
    for i, symptom in enumerate(feature_list):
        if symptom in matched_symptoms:
            input_vector[i] = 1
    return input_vector

# -----------------------------
# Main prediction interface
# -----------------------------
def main():
    print("\nWelcome to the Disease Prediction CLI!")
    print("Please describe your symptoms in detail.\n")

    # Get user input
    user_input = input("Enter your symptoms: ")

    # Use GPT-4o mini to interpret the symptoms
    matched_symptoms = interpret_symptoms_with_gpt(user_input, feature_cols)

    if not matched_symptoms:
        print("Could not interpret the symptoms provided. Please try again.")
        return

    print(f"\nInterpreted symptoms: {', '.join(matched_symptoms)}\n")

    # Convert matched symptoms to binary vector and then to a DataFrame to preserve feature names.
    input_vector = symptoms_to_vector(matched_symptoms, feature_cols).reshape(1, -1)
    input_df = pd.DataFrame(input_vector, columns=feature_cols)

    # Predict the disease using the pre-trained model.
    predicted_label = model.predict(input_df)[0]

    # Convert predicted label back to disease name.
    predicted_disease = uniques[predicted_label]

    # Retrieve recommended treatment.
    treatment_recommendation = get_treatment(predicted_disease)

    # Display prediction and treatment.
    print("\nPrediction Results:")
    print("-------------------")
    print(f"Predicted Disease: {predicted_disease}")
    print(f"Recommended Treatment: {treatment_recommendation}")

if __name__ == "__main__":
    main()
