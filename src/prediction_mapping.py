import pandas as pd
from rapidfuzz import process, fuzz
from data_utils import load_and_clean_data

# ============================================================
# File: prediction_mapping.py
# Purpose: Map predicted disease labels to corresponding treatments.
#          This uses fuzzy matching with RapidFuzz so that if the
#          disease name isn’t an exact match, we can still find the
#          closest treatment.
# ============================================================

# -----------------------------
# Define file path for the Diseases_Symptoms dataset
# -----------------------------
disease_symptoms_path = "../data/Diseases_Symptoms.csv"

# -----------------------------
# Load and clean the Diseases_Symptoms dataset
# -----------------------------
disease_symptoms_df = load_and_clean_data(disease_symptoms_path, 'diseases_symptoms')

# -----------------------------
# Build a mapping from disease names to treatments
# -----------------------------
disease_to_treatment = dict(zip(disease_symptoms_df['Name'], disease_symptoms_df['Treatments']))


# -----------------------------
# Define a fuzzy matching function to retrieve treatment recommendations
# -----------------------------
def get_treatment(disease_name, threshold=80):
    """
    Given a disease name, return the corresponding treatment(s).
    Uses fuzzy matching (via RapidFuzz) if an exact match is not found.

    Parameters:
      disease_name (str): The name of the predicted disease.
      threshold (int): Minimum fuzzy matching score (0-100) to consider a match.

    Returns:
      str: Treatment recommendation or a message if not available.
    """
    # First try an exact match
    if disease_name in disease_to_treatment:
        return disease_to_treatment[disease_name]

    # If not found, use fuzzy matching to find the best match among the keys
    best_match, score, _ = process.extractOne(disease_name, disease_to_treatment.keys(), scorer=fuzz.ratio)
    if score >= threshold:
        return disease_to_treatment[best_match]
    else:
        return "Treatment information not available"


# -----------------------------
# Example usage: Testing the mapping with sample diseases
# -----------------------------
if __name__ == "__main__":
    sample_diseases = [
        "Fungal infection of the skin",
        "Insulin Overdose",
        "Bipolar Disorder",
        "Cellulitis",
        "Non-existent Disease"  # To test default behavior
    ]

    for disease in sample_diseases:
        treatment = get_treatment(disease)
        print(f"Disease: {disease}")
        print(f"Recommended Treatment: {treatment}\n")
