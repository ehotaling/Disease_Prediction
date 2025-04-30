import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained model
# from transformers import T5ForConditionalGeneration, T5Tokenizer # Not used anymore
from data_utils import load_and_clean_data
import os
from openai import OpenAI
from dotenv import load_dotenv
import torch
import torch.nn as nn
import sys # For exit

# ============================================================
# File: predict_cli.py
# Purpose: Provide a command-line interface (CLI) for making predictions.
#          Uses OpenAI API (GPT-4o mini) to interpret user-input symptoms,
#          loads a selected pre-trained model (RF, LR, or MLP),
#          predicts the disease, and generates treatment recommendations via OpenAI API.
# ============================================================

# -----------------------------
# Configuration and Paths
# -----------------------------
rf_model_path = "../models/rf_model.pkl"
lr_model_path = "../models/lr_model.pkl"
mlp_model_path = "../models/mlp_model.pth"
training_data_path = "../data/training_data.csv"


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
# Wrap data loading in try-except at the start
try:
    print("Loading and cleaning data...")
    df = load_and_clean_data(training_data_path, 'training')
    feature_cols = list(df.columns.drop('prognosis'))
    # Factorize the target column to get the mapping of integer labels to disease names.
    y_encoded, uniques = pd.factorize(df['prognosis'])
    print(f"Data loaded. Found {len(uniques)} unique classes (diseases).")

    # For symptom extraction
    features_str = ", ".join(feature_cols)
    EXTRACTION_INSTRUCTIONS = (
            "You are a medical symptom extraction assistant.  "
            "These are the only valid symptom names: " + features_str + ".  "
            "When I give you user text, return ONLY a single‐line, comma‐separated list "
            "of the EXACT matching symptoms.  Do NOT include any explanations, numbers, bullets, "
            "newlines or extra text.  JUST the list."
    )

    # print("Target classes (label mapping):", uniques.tolist()) # Can be very long
except FileNotFoundError:
    print(f"FATAL ERROR: Training data file not found at {training_data_path}")
    sys.exit(1) # Exit if data cannot be loaded
except Exception as e:
    print(f"FATAL ERROR: Could not load or process training data: {e}")
    sys.exit(1)

# -----------------------------
# Initialize OpenAI API Client
# -----------------------------
try:
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    client = OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized.")
    prev_id = None  # Tracks the first API response ID to enable stateful chaining

except ValueError as e:
    print(f"FATAL ERROR: {e}")
    print("Please ensure your .env file exists and contains the OPENAI_API_KEY.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: Could not initialize OpenAI client: {e}")
    sys.exit(1)


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


# -----------------------------
# Symptom Interpretation Function
# -----------------------------
def interpret_symptoms_with_gpt(user_input):
    """
    Uses the Responses API to extract only known symptoms using conversation state,
    then logs exactly which extracted tokens matched the feature list.
    """
    global prev_id
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            previous_response_id=prev_id,
            input=user_input,
            instructions = EXTRACTION_INSTRUCTIONS
        )

        # Pull out the raw text
        extracted_text = ""
        if (response.output
            and response.output[0].content
            and response.output[0].content[0].text):
            extracted_text = response.output[0].content[0].text.strip()
        else:
            print("Warning: No output returned from GPT.")
            return []

        # DEBUG: what GPT actually gave you
        print(f"\n[DEBUG] GPT raw extraction:\n{extracted_text}\n")

        # Split on commas – normalize underscores/spaces, lowercase
        tokens = [
            tok.strip().lower().replace('_', ' ')
            for tok in extracted_text.split(',')
            if tok.strip()
        ]

        # Normalize feature list once
        normalized_features = {
            feat.strip().lower().replace('_', ' ')
            for feat in feature_cols
        }

        # Partition extracted tokens into matched vs unmatched
        matched = [tok for tok in tokens if tok in normalized_features]
        unmatched = [tok for tok in tokens if tok not in normalized_features]

        # Print explicit mapping
        print(f"Matched {len(matched)} token(s): {matched}")
        if unmatched:
            print(f"Unmatched token(s) — not in feature list: {unmatched}")

        return matched

    except Exception as e:
        print(f"Error extracting symptoms via GPT: {e}")
        return []




# -----------------------------
# Treatment Recommendation Function (Using new API - client.responses.create)
# -----------------------------
def get_treatment_recommendation_gpt(disease_name):
    """
    Uses GPT-4o mini via client.responses.create to generate a concise treatment recommendation.
    Includes a disclaimer about not being professional medical advice.

    Parameters:
        disease_name (str): The name of the predicted disease.

    Returns:
        str: AI-generated treatment recommendation or an error message.
    """
    prompt = (
        f"Provide a concise summary of typical management or treatment approaches for '{disease_name}'. "
        f"Focus on general information. Do not give specific medical advice."
    )
    instructions_text = (
        "You are a helpful assistant providing general information about medical conditions. "
        "Summarize typical treatment approaches concisely. Start directly with the information. "
        "IMPORTANT: Explicitly state that this information is not a substitute for professional medical advice."
    )

    try:
        print(f"Getting treatment recommendation for '{disease_name}' via OpenAI API...")
        response = client.responses.create(
            model="gpt-4o-mini",
            previous_response_id=prev_id,
            input=prompt,
            instructions=instructions_text,
            max_output_tokens=250
        )

        # Extract text
        if response.output and len(response.output) > 0 and \
           response.output[0].content and len(response.output[0].content) > 0 and \
           response.output[0].content[0].text:

            recommendation = response.output[0].content[0].text
            # Add disclaimer if not already included by the model (though prompt asks it to)
            disclaimer = "Disclaimer: This is AI-generated information and NOT a substitute for professional medical advice. Consult a healthcare provider."
            if "disclaimer" not in recommendation.lower() and "medical advice" not in recommendation.lower():
                 recommendation += f"\n\n{disclaimer}"
            print("API treatment recommendation received.")
            return recommendation
        else:
            print("Warning: Could not extract treatment text from API response structure.")
            print(f"Full API Response Status: {response.status}")
            return "Treatment information could not be generated."

    except Exception as e:
        print(f"Error communicating with OpenAI API for treatment recommendation: {e}")
        return "Error generating treatment recommendation."


# -----------------------------
# Symptom to Vector Function
# -----------------------------
def symptoms_to_vector(matched_symptoms, feature_list):
    """Converts matched symptoms to a binary vector after normalizing both sides."""
    normalized_input = [s.strip().lower().replace('_', ' ') for s in matched_symptoms]
    normalized_features = [f.strip().lower().replace('_', ' ') for f in feature_list]

    input_vector = np.zeros(len(feature_list), dtype=int)
    for i, feature in enumerate(normalized_features):
        if feature in normalized_input:
            input_vector[i] = 1
    return input_vector

# -----------------------------
# PyTorch MLP Model Definition
# -----------------------------
# This class defines the structure of the Multi-Layer Perceptron (MLP)
# used for classification. It must be identical to the MLPClassifier
# class defined in model_training.py to allow loading the saved model state_dict.
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Hidden layer with 64 neurons
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(64, num_classes)  # Output layer for class logits

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def initialize_gpt_symptom_session(feature_list):
    """
    Sends the feature list once to the OpenAI responses API to start a stateful session.
    Returns the response ID for chaining.
    """
    try:
        features_str = ", ".join(feature_list)

        response = client.responses.create(
            model="gpt-4o-mini",
            input="Session start.",
            instructions=EXTRACTION_INSTRUCTIONS
        )

        return response.id

    except Exception as e:
        print(f"Error initializing GPT session: {e}")
        sys.exit(1)

# -----------------------------
# Main prediction interface
# -----------------------------
def main():
    """Main CLI loop for symptom input, prediction, and treatment recommendation."""
    print("\nWelcome to the Disease Prediction CLI!\n")

    global prev_id
    print("Initializing GPT conversation context with known symptom list...")
    prev_id = initialize_gpt_symptom_session(feature_cols)
    if not prev_id:
        print("Failed to initialize conversation context.")
        sys.exit(1)


    # --- Model Selection ---
    print("Available models: RF (Random Forest), LR (Logistic Regression), MLP (Neural Network)")
    while True:
        choice = input("Enter the model you want to use (RF/LR/MLP): ").upper()
        if choice in ['RF', 'LR', 'MLP']:
            break
        else:
            print("Invalid choice. Please enter RF, LR, or MLP.")

    selected_model = None
    model_name = ""

    # --- Model Loading ---
    try:
        print(f"Loading model '{choice}'...")
        if choice == 'RF':
            selected_model = joblib.load(rf_model_path)
            model_name = "Random Forest"
        elif choice == 'LR':
            selected_model = joblib.load(lr_model_path)
            model_name = "Logistic Regression"
        elif choice == 'MLP':
            model_name = "PyTorch MLP"
            try:
                input_dim = len(feature_cols)
                num_classes = len(uniques)
                mlp_model_instance = MLPClassifier(input_dim, num_classes)
                # Load state dict - ensure model is on CPU if saved from GPU and running on CPU now, or vice versa
                # Determine map_location based on current availability vs where it might have been saved
                map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                mlp_model_instance.load_state_dict(torch.load(mlp_model_path, map_location=map_location))
                mlp_model_instance.eval()
                selected_model = mlp_model_instance
                print(f"{model_name} weights loaded successfully.")
            except FileNotFoundError:
                print(f"Error: MLP model file not found at {mlp_model_path}")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading PyTorch MLP model: {e}")
                sys.exit(1)

        if selected_model:
            print(f"{model_name} model loaded successfully.")
        # Add check if model failed to load (e.g., MLP case)
        elif choice == 'MLP' and not selected_model:
             print(f"Error: Failed to load {model_name}. Exiting.")
             sys.exit(1)


    except FileNotFoundError:
        print(f"Error: Model file for {choice} not found. Please ensure models are trained and saved.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading the model: {e}")
        sys.exit(1)

    # --- Main Prediction Loop ---
    while True:
        print("\nPlease describe your symptoms in detail (or type 'quit' to exit).")
        user_input = input("Enter your symptoms: ")
        if user_input.lower() == 'quit':
             break

        # 1. Interpret Symptoms
        matched_symptoms = interpret_symptoms_with_gpt(user_input)

        if not matched_symptoms:
            print("Could not interpret valid symptoms from input. Please try again or rephrase.")
            continue # Ask for input again
        else:
            print(f"\nInterpreted symptoms: {', '.join(matched_symptoms)}")

            # 2. Convert to Vector
            input_vector = symptoms_to_vector(matched_symptoms, feature_cols).reshape(1, -1)

            # DEBUG: show exactly which features went into the model
            active_idxs = [i for i, bit in enumerate(input_vector[0]) if bit == 1]
            print(f"[DEBUG] Active feature indices: {active_idxs}")
            print(f"[DEBUG] Active feature names: {[feature_cols[i] for i in active_idxs]}\n")

            input_df = pd.DataFrame(input_vector, columns=feature_cols)  # For sklearn models

            # DEBUG: Show top 5 candidate diseases
            probs = selected_model.predict_proba(input_df)[0]
            top5 = np.argsort(probs)[-5:][::-1]
            print("Top 5 candidate diseases:")
            for idx in top5:
                print(f"  {uniques[idx]} ({probs[idx] * 100:.1f} %)")



            # 3. Predict Disease
            predicted_label = None
            predicted_disease = None
            try:
                if choice in ['RF', 'LR']:
                    if selected_model:
                        predicted_label = selected_model.predict(input_df)[0]
                    else:
                        print(f"Error: {model_name} model not loaded.")
                elif choice == 'MLP':
                    if selected_model:
                         # Determine device for inference
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        selected_model.to(device) # Ensure model is on the correct device
                        input_tensor = torch.tensor(input_vector.astype(np.float32)).float().to(device)
                        with torch.no_grad():
                            outputs = selected_model(input_tensor)
                            _, predicted = torch.max(outputs.data, 1)
                            predicted_label = predicted.item()
                    else:
                        print("Error: MLP model not loaded.")

            except Exception as e:
                print(f"Error during model prediction: {e}")
                predicted_label = None

            # 4. Get Disease Name
            if predicted_label is not None:
                try:
                    predicted_disease = uniques[predicted_label]
                except IndexError:
                    print(f"Error: Predicted label {predicted_label} is out of bounds.")
                    predicted_disease = None
            else:
                 predicted_disease = None

            # 5. Get Treatment Recommendation (if disease predicted)
            treatment_recommendation = "N/A" # Default
            if predicted_disease:
                 treatment_recommendation = get_treatment_recommendation_gpt(predicted_disease)


            # 6. Display Results
            print("\n--- Prediction Results ---")
            print(f"Model Used: {model_name}")
            if predicted_disease:
                print(f"Predicted Disease: {predicted_disease}")
                print(f"Recommended Treatment Info:\n{treatment_recommendation}")
            else:
                print("Could not predict disease based on input.")
            print("--------------------------")

        # Ask if user wants to enter another set of symptoms
        again = input("\nWould you like to enter another set of symptoms? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            # print("Thank you for using the Disease Prediction CLI. Stay healthy!")
            break # Exit the inner loop

    print("\nThank you for using the Disease Prediction CLI. Stay healthy!")


if __name__ == "__main__":
    main()