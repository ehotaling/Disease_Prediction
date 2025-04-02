
from data_utils import load_and_clean_data


# -----------------------------
# Load and prepare the Diseases_Symptoms dataset
# -----------------------------
disease_symptoms_path = "../data/Diseases_Symptoms.csv"
disease_symptoms_df = load_and_clean_data(disease_symptoms_path, 'diseases_symptoms')

# Build a mapping from canonical disease names to treatments
disease_to_treatment = dict(zip(disease_symptoms_df['Name'], disease_symptoms_df['Treatments']))


def get_treatment(predicted_disease):
    """
    Given a predicted disease name (which is expected to be in canonical form),
    return the corresponding treatment via a direct lookup.

    Parameters:
      predicted_disease (str): The disease name predicted by the model.

    Returns:
      str: Treatment recommendation or a message if not available.
    """
    return disease_to_treatment.get(predicted_disease, "Treatment information not available")


# -----------------------------
# Example usage: Testing the mapping with a sample predicted disease.
# -----------------------------
if __name__ == "__main__":
    sample_disease = "Gastroenteritis"  # Expected canonical form from normalized training data
    treatment = get_treatment(sample_disease)
    print(f"Predicted Disease: {sample_disease}")
    print(f"Recommended Treatment: {treatment}")
