import pandas as pd
from data_utils import load_and_clean_data

# ============================================================
# File: prediction_mapping.py
# Purpose: Map predicted disease labels to corresponding treatments.
#          This simulates the step of taking a model's output and
#          providing a recommended treatment.
# ============================================================

# -----------------------------
# Define file path for the Diseases_Symptoms dataset
# -----------------------------
# This CSV file contains disease names along with their associated symptoms and treatments.
disease_symptoms_path = "../data/Diseases_Symptoms.csv"

# -----------------------------
# Load and clean the Diseases_Symptoms dataset
# -----------------------------
# We use our common utility function to ensure consistent cleaning.
disease_symptoms_df = load_and_clean_data(disease_symptoms_path, 'diseases_symptoms')

# -----------------------------
# Build a mapping from disease names to treatments
# -----------------------------
# Using pandas' zip function, we create a dictionary where each disease name (from the 'Name' column)
# maps to its recommended treatment (from the 'Treatments' column).
disease_to_treatment = dict(zip(disease_symptoms_df['Name'], disease_symptoms_df['Treatments']))


# -----------------------------
# Define a function to retrieve treatment recommendations
# -----------------------------
def get_treatment(disease_name):
    """
    Given a disease name, return the corresponding treatment(s).
    If the disease is not found in our mapping, return a default message.

    Parameters:
      disease_name (str): The name of the predicted disease.

    Returns:
      str: Treatment recommendation or a message if not available.
    """
    return disease_to_treatment.get(disease_name, "Treatment information not available")


# -----------------------------
# Example usage: Testing the mapping with sample diseases
# -----------------------------
if __name__ == "__main__":
    # Define some sample diseases to test the mapping.
    sample_diseases = [
        "Fungal infection",
        "Allergy",
        "GERD",
        "Non-existent Disease"  # This will test the default message.
    ]

    # Loop through the sample diseases, retrieve and print the treatment for each.
    for disease in sample_diseases:
        treatment = get_treatment(disease)
        print(f"Disease: {disease}")
        print(f"Recommended Treatment: {treatment}\n")
