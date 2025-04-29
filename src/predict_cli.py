import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained model
from rapidfuzz import process, fuzz
from transformers import T5ForConditionalGeneration, T5Tokenizer
from data_utils import load_and_clean_data
from prediction_mapping import get_treatment
import os
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# File: predict_cli.py
# Purpose: Provide a command-line interface (CLI) for making predictions.
#          Uses OpenAI API to interpret user-input symptoms,
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
# Load T5 model for symptom interpretation
# -----------------------------
# t5_model_name = 't5-base' # Or 't5-base', etc.
# t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
# t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
# print(f"T5 model ({t5_model_name}) loaded successfully.")

# -----------------------------
# Load training data to extract feature names and target mapping
# -----------------------------
df = load_and_clean_data(training_data_path, 'training')
feature_cols = list(df.columns.drop('prognosis'))

# Factorize the target column to get the mapping of integer labels to disease names.
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


# def interpret_symptoms_with_t5(user_input, feature_list, tokenizer, model):
#     """
#         Interprets medical symptoms from a user input string using a T5 model. The function generates a
#         comma-separated list of symptoms identified in the input text and matches these symptoms against
#         a predefined feature list.
#         """
#     prompt = f"extract medical symptom keywords from the following text: \"{user_input}\"" # Experiment with prefixes like "find symptoms:" etc.
#
#     try:
#         input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids # Ensure truncation happens if needed
#         outputs = model.generate(input_ids, max_length=100) # Adjust output length as needed
#         decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#         # Extract potential symptoms (e.g., split by comma, space, or newline depending on T5 output format)
#         # Example: simple comma splitting, might need refinement
#         potential_symptoms = [s.strip().lower() for s in decoded_output.split(',')] # Or better splitting logic
#
#         # 1. Find direct matches
#         matched_symptoms = {symptom for symptom in potential_symptoms if
#                             symptom in feature_list}  # Use a set for efficiency
#         unmatched_potentials = [p for p in potential_symptoms if p not in matched_symptoms]
#
#         # 2. Try fuzzy matching *only* for terms that didn't match directly
#         for potential in unmatched_potentials:
#             match, score, _ = process.extractOne(potential, feature_list, scorer=fuzz.WRatio)
#             if score > 80:  # Adjust threshold
#                 # Add the matched feature if it wasn't already found directly
#                 matched_symptoms.add(match)  # Add to the set
#
#         return list(matched_symptoms)  # Convert back to list
#
#     except Exception as e:
#         print(f"Error during T5 symptom interpretation: {e}")
#         return []


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
    Converts a list of matched symptoms into a binary vector representation based on a given feature list. Each position
    of the vector corresponds to a symptom in the feature list. If a symptom is present in the matched symptoms, the
    corresponding position in the binary vector is set to 1; otherwise, it remains 0.

    :param matched_symptoms: List of symptoms that have been identified or matched.
    :type matched_symptoms: list of str
    :param feature_list: List of all possible symptoms (features) that define the vector space.
    :type feature_list: list of str
    :return: A binary vector of integers, where each position indicates the presence or absence of a symptom from the
        feature list.
    :rtype: numpy.ndarray
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
    """
    Provides a command-line interface (CLI) for disease prediction based on
    user-provided symptoms, leveraging machine learning models for interpretation
    and recommendation of treatments.

    The CLI accepts symptoms from the user in textual format, interprets these
    symptoms using a symptom interpretation model, predicts the disease using
    a pre-trained classifier, and then provides a corresponding treatment
    recommendation. Users can perform multiple predictions within the same session.

    """
    print("\nWelcome to the Disease Prediction CLI!")

    while True:
        print("\nPlease describe your symptoms in detail.")
        user_input = input("Enter your symptoms: ")

        # Use GPT-4o mini to interpret the symptoms
        matched_symptoms = interpret_symptoms_with_gpt(user_input, feature_cols)

        # Use a T5 model to interpret the symptoms
        # matched_symptoms = interpret_symptoms_with_t5(user_input, feature_cols, t5_tokenizer, t5_model)

        if not matched_symptoms:
            print("Could not interpret the symptoms provided. Please try again.")
        else:
            print(f"\nInterpreted symptoms: {', '.join(matched_symptoms)}\n")

            # Convert matched symptoms to a binary vector and then to a DataFrame
            input_vector = symptoms_to_vector(matched_symptoms, feature_cols).reshape(1, -1)
            input_df = pd.DataFrame(input_vector, columns=feature_cols)

            # Predict the disease using the pre-trained model
            predicted_label = model.predict(input_df)[0]
            predicted_disease = uniques[predicted_label]

            # Retrieve recommended treatment
            treatment_recommendation = get_treatment(predicted_disease)

            # Display prediction and treatment
            print("\nPrediction Results:")
            print("-------------------")
            print(f"Predicted Disease: {predicted_disease}")
            print(f"Recommended Treatment: {treatment_recommendation}")

        # Ask if the user wants to enter another set of symptoms
        again = input("\nWould you like to enter another set of symptoms? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            print("Thank you for using the Disease Prediction CLI. Stay healthy!")
            break


if __name__ == "__main__":
    main()