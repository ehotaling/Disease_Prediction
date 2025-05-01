import pandas as pd
import numpy as np
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv
import torch
import torch.nn as nn
import sys
import numpy as np

# Import necessary functions from local utility script
from data_utils import load_and_clean_data

# ============================================================
# File: predict_cli.py
# Purpose: Provide a command-line interface (CLI) for making predictions.
#          Uses OpenAI API (GPT-4o mini) for stateless symptom interpretation
#          and treatment recommendation generation. Loads a selected pre-trained
#          model (RF, LR, or MLP) for disease prediction.
# ============================================================

# -----------------------------
# Configuration and Paths
# -----------------------------
# Define relative paths to saved model files and training data
rf_model_path = "../models/rf_model.pkl"
lr_model_path = "../models/lr_model.pkl"
mlp_model_path = "../models/mlp_model.pth"
training_data_path = "../data/training_data.csv"

# -----------------------------
# Global Constants
# -----------------------------
# Instructions for the OpenAI API call for symptom extraction.
# This template will be formatted with the actual feature list after loading data.
EXTRACTION_INSTRUCTIONS_TEMPLATE = (
    "You are a medical symptom extraction assistant. "
    "Identify symptom keywords mentioned in the user's text that EXACTLY MATCH one of the following valid symptom names: {feature_string}. "
    "Return ONLY a single‐line, comma‐separated list of the EXACT matching valid symptom names found in the text. "
    "Do NOT include symptoms not on the list. Do NOT include explanations, numbers, bullets, newlines or extra text. JUST the list."
)
# Global variable to hold the fully formatted extraction instructions
EXTRACTION_INSTRUCTIONS = ""

# Instructions for the OpenAI API call for treatment recommendation.
TREATMENT_INSTRUCTIONS = (
    "You are a helpful assistant providing general information about medical conditions. "
    "Summarize typical treatment approaches concisely for the given disease name. Start directly with the information. "
    "IMPORTANT: Explicitly state that this information is not a substitute for professional medical advice."
)

# -----------------------------
# Load training data (once at the start) and Prepare Global Variables
# -----------------------------
try:
    print("Loading and cleaning data (needed for feature columns)...")
    # Load data just to get column names, we don't need y_encoded from here anymore
    df = load_and_clean_data(training_data_path, 'training')
    feature_cols = list(df.columns.drop('prognosis'))
    print(f"Data loaded. Using {len(feature_cols)} features.")

    # --- Load the label mapping saved during training ---
    label_map_path = "../models/label_mapping.npy"
    print(f"Loading label mapping from {label_map_path}...")
    uniques = np.load(label_map_path, allow_pickle=True) # Load the saved uniques array
    print(f"Loaded mapping for {len(uniques)} unique classes (diseases).")
    # --- End label mapping load ---

    # --- Construct the full extraction instructions string ---
    features_str = ", ".join(f"'{feat}'" for feat in feature_cols)
    EXTRACTION_INSTRUCTIONS = EXTRACTION_INSTRUCTIONS_TEMPLATE.format(feature_string=features_str)
    print(f"Length of extraction instructions: {len(EXTRACTION_INSTRUCTIONS)} characters.")
    if len(EXTRACTION_INSTRUCTIONS) > 15000:
         print("Warning: Extraction instructions are very long, may approach API limits.")

# Remove the old factorization line:
# y_encoded, uniques = pd.factorize(df['prognosis']) # REMOVE THIS LINE

except FileNotFoundError as e:
    # Handle error if label mapping file is missing
    if 'label_mapping.npy' in str(e):
         print(f"FATAL ERROR: Label mapping file not found at {label_map_path}")
         print("Please ensure model_training.py has been run successfully.")
    else:
         print(f"FATAL ERROR: Training data file not found at {training_data_path}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: Could not load data or label mapping: {e}")
    sys.exit(1)

# -----------------------------
# Initialize OpenAI API Client
# -----------------------------
try:
    # Load environment variables from .env file
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # Ensure the API key is found
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    # Create the OpenAI client instance
    client = OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized.")
except ValueError as e:
    print(f"FATAL ERROR: {e}")
    print("Please ensure your .env file exists and contains the OPENAI_API_KEY.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: Could not initialize OpenAI client: {e}")
    sys.exit(1)


# -----------------------------
# Symptom Interpretation Function
# -----------------------------
def interpret_symptoms_with_gpt(user_input):
    """
    Uses OpenAI API (stateless call) to extract symptom keywords based on
    globally defined instructions containing the valid feature list.
    Assumes the API returns only valid, comma-separated symptoms.
    """
    # The user's raw text is the primary input to the model
    prompt = user_input
    try:
        print("Interpreting symptoms via OpenAI API (using full list in instructions)...")
        # Make the API call using the 'responses' endpoint
        response = client.responses.create(
            model="gpt-4o-mini", # Specify the desired OpenAI model
            input=prompt, # User's symptom description
            instructions=EXTRACTION_INSTRUCTIONS # Pre-formatted instructions including feature list
        )

        # NEW: print raw response for debug
        print(f"[DEBUG] Full raw response:\n{response}")
        # Initialize extracted text
        extracted_text = ""
        # Safely access the nested text output from the response object
        if (response.output and response.output[0].content and response.output[0].content[0].text):
            extracted_text = response.output[0].content[0].text.strip()
        else:
            print("Warning: No output text returned from GPT for symptom interpretation.")
            return [] # Return empty list if no text found

        print(f"\n[DEBUG] GPT raw extraction (expected valid symptoms only):\n{extracted_text}\n")

        # Process the raw text: split by comma, normalize (lowercase, strip, replace underscore)
        # Assumes the API followed instructions perfectly.
        matched_symptoms = [
            tok.strip().lower().replace('_', ' ')
            for tok in extracted_text.split(',')
            if tok.strip() # Ignore empty strings resulting from split
        ]

        return matched_symptoms

    except Exception as e:
        # Catch any exceptions during the API call or processing
        print(f"Error extracting symptoms via GPT: {e}")
        import traceback
        traceback.print_exc() # Print detailed error for debugging
        return [] # Return empty list on error


# -----------------------------
# Treatment Recommendation Function
# -----------------------------
def get_treatment_recommendation_gpt(disease_name):
    """
    Uses OpenAI API (stateless call) to generate a concise treatment recommendation
    for the given disease name. Includes a disclaimer.
    """
    # Create a prompt asking for general treatment information for the specific disease
    prompt = (
        f"Provide a concise summary of typical management or treatment approaches for '{disease_name}'. "
        f"Focus on general information. Do not give specific medical advice."
    )
    try:
        print(f"Getting treatment recommendation for '{disease_name}' via OpenAI API...")
        # Make the stateless API call using the 'responses' endpoint
        response = client.responses.create(
            model="gpt-4o-mini", # Specify the model
            input=prompt, # The request for treatment info
            instructions=TREATMENT_INSTRUCTIONS, # Specific instructions for this task
            max_output_tokens=250 # Limit response length for conciseness
        )

        # Default message if extraction fails
        recommendation = "Treatment information could not be generated."
        # Safely extract the generated text
        if (response.output and response.output[0].content and response.output[0].content[0].text):
            recommendation = response.output[0].content[0].text.strip()
            # Define the disclaimer text
            disclaimer = "Disclaimer: This is AI-generated information and NOT a substitute for professional medical advice. Consult a healthcare provider."
            # Append disclaimer if the model didn't include it
            if "disclaimer" not in recommendation.lower() and "medical advice" not in recommendation.lower():
                 recommendation += f"\n\n{disclaimer}"
            print("API treatment recommendation received.")
        else:
            # Handle cases where the API response structure is unexpected
            print("Warning: Could not extract treatment text from API response structure.")
            print(f"Full API Response Status: {response.status}")

        return recommendation

    except Exception as e:
        # Catch exceptions during the API call
        print(f"Error communicating with OpenAI API for treatment recommendation: {e}")
        return "Error generating treatment recommendation."


# -----------------------------
# Symptom to Vector Function
# -----------------------------
def symptoms_to_vector(matched_symptoms, feature_list):
    """
    Converts a list of matched symptom strings into a binary numpy vector
    based on the master feature_list order. Normalizes both lists for matching.
    """
    # Normalize the matched symptoms received (lowercase, strip, space for underscore)
    normalized_input_set = {s.strip().lower().replace('_', ' ') for s in matched_symptoms}

    # Initialize the binary vector with zeros
    input_vector = np.zeros(len(feature_list), dtype=int)
    # Iterate through the original feature list to maintain correct order
    for i, feature in enumerate(feature_list):
        # Normalize the feature from the master list for comparison
        normalized_feature = feature.strip().lower().replace('_', ' ')
        # If the normalized feature is in the set of matched symptoms, set vector position to 1
        if normalized_feature in normalized_input_set:
            input_vector[i] = 1
    return input_vector

# -----------------------------
# PyTorch MLP Model Definition
# -----------------------------
# Defines the MLP architecture; must match the structure used during training
# to correctly load the saved weights (state_dict).
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        # Define layers: Input -> Hidden (64 neurons) -> ReLU -> Output
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Define the forward pass logic
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# -----------------------------
# Main prediction interface
# -----------------------------
def main():
    """
    Main function to run the interactive Command Line Interface (CLI).
    Handles model selection, loading, symptom input, interpretation,
    prediction, treatment recommendation, and user interaction loop.
    """
    print("\nWelcome to the Disease Prediction CLI!\n")

    # --- Model Selection ---
    # Prompt user to choose which trained model to use
    print("Available models: RF (Random Forest), LR (Logistic Regression), MLP (Neural Network)")
    while True: # Loop until valid input is received
        choice = input("Enter the model you want to use (RF/LR/MLP): ").upper()
        if choice in ['RF', 'LR', 'MLP']:
            break # Exit loop if choice is valid
        else:
            print("Invalid choice. Please enter RF, LR, or MLP.")

    # Initialize variables for the selected model and its name
    selected_model = None
    model_name = ""

    # --- Model Loading ---
    # Attempt to load the chosen model file
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
                # Get necessary dimensions from loaded data info
                input_dim = len(feature_cols)
                num_classes = len(uniques)
                # Instantiate the MLP class structure
                mlp_model_instance = MLPClassifier(input_dim, num_classes)
                # Determine device (CPU/GPU) for loading
                map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Load the saved weights (state dictionary) onto the determined device
                # Load full checkpoint (state_dict + metadata)
                checkpoint = torch.load(mlp_model_path, map_location=map_location)
                # OPTIONAL: Validate input_dim and num_classes match expectations
                if checkpoint.get('input_dim') != input_dim or checkpoint.get('num_classes') != num_classes:
                    print("Warning: MLP input dimensions or number of classes do not match current config.")
                mlp_model_instance.load_state_dict(checkpoint['model_state_dict'])

                # Set the model to evaluation mode (disables dropout, etc.)
                mlp_model_instance.eval()
                # Assign the loaded instance
                selected_model = mlp_model_instance
                print(f"{model_name} weights loaded successfully.")
            except FileNotFoundError:
                print(f"Error: MLP model file not found at {mlp_model_path}")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading PyTorch MLP model: {e}")
                sys.exit(1)

        # Confirm successful loading or handle MLP loading failure
        if selected_model:
            print(f"{model_name} model loaded successfully.")
        elif choice == 'MLP' and not selected_model:
             print(f"Error: Failed to load {model_name}. Exiting.")
             sys.exit(1)

    # Handle errors if model files are not found
    except FileNotFoundError as e:
        print(f"Error: Model file not found ({e}). Please ensure models are trained and saved.")
        sys.exit(1)
    # Catch any other unexpected errors during loading
    except Exception as e:
        print(f"An unexpected error occurred loading the model: {e}")
        sys.exit(1)

    # --- Main Prediction Loop ---
    # Loop to allow multiple symptom inputs per session
    while True:
        print("\nPlease describe your symptoms in detail (or type 'quit' to exit).")
        user_input = input("Enter your symptoms: ")
        # Allow user to quit
        if user_input.lower() == 'quit':
             break

        # 1. Interpret Symptoms using OpenAI API
        matched_symptoms = interpret_symptoms_with_gpt(user_input)

        # If no valid symptoms are returned by the API, prompt again
        if not matched_symptoms:
            print("Could not interpret valid symptoms from input. Please check API response or try again.")
            continue # Go to the next iteration of the loop
        else:
            # If symptoms are found, proceed with prediction
            print(f"\nInterpreted symptoms (from API): {', '.join(matched_symptoms)}")

            # 2. Convert matched symptoms to a binary feature vector
            input_vector = symptoms_to_vector(matched_symptoms, feature_cols).reshape(1, -1)

            # --- DEBUG BLOCKS ---
            # Display which features are active in the vector
            active_idxs = [i for i, bit in enumerate(input_vector[0]) if bit == 1]
            print(f"[DEBUG] Active feature indices: {active_idxs}")
            print(f"[DEBUG] Active feature names: {[feature_cols[i] for i in active_idxs]}\n")

            # Display top 5 likely diseases for RF/LR models if possible
            if choice in ['RF', 'LR']:
                 if hasattr(selected_model, "predict_proba"): # Check if model supports probabilities
                     try:
                         # Create DataFrame temporarily for predict_proba input
                         input_df_debug = pd.DataFrame(input_vector, columns=feature_cols)
                         # Get probabilities for all classes
                         probs = selected_model.predict_proba(input_df_debug)[0]
                         # Get indices of top 5 probabilities
                         top5 = np.argsort(probs)[-5:][::-1]
                         print("Top 5 candidate diseases (Probabilities):")
                         for idx in top5:
                              if idx < len(uniques): # Check index bounds
                                  print(f"  {uniques[idx]} ({probs[idx] * 100:.1f} %)")
                              else:
                                  print(f"  Invalid index {idx}")
                         print("-" * 20) # Separator
                     except Exception as e:
                         print(f"[DEBUG] Could not get probabilities: {e}")
                 else:
                     print(f"[DEBUG] {model_name} model does not have predict_proba method.")
            # --- END DEBUG ---

            # 3. Predict Disease using the loaded model
            predicted_label = None
            predicted_disease = None
            try:
                # Use .predict() for scikit-learn models
                if choice in ['RF', 'LR']:
                    if selected_model:
                        # Create DataFrame required for sklearn predict input
                        input_df = pd.DataFrame(input_vector, columns=feature_cols)
                        predicted_label = selected_model.predict(input_df)[0]
                    else:
                        print(f"Error: {model_name} model not loaded.")
                # Use forward pass for PyTorch MLP model
                elif choice == 'MLP':
                    if selected_model:
                         # Determine device (CPU/GPU)
                         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                         # Ensure model is on the correct device
                         selected_model.to(device)
                         # Convert input vector to tensor and move to device
                         input_tensor = torch.tensor(input_vector.astype(np.float32)).float().to(device)
                         # Perform inference without calculating gradients
                         with torch.no_grad():
                             outputs = selected_model(input_tensor)
                             # Get the index of the highest probability class
                             _, predicted = torch.max(outputs.data, 1)
                             predicted_label = predicted.item()
                    else:
                        print("Error: MLP model not loaded.")

            except Exception as e:
                # Catch errors during the prediction step
                print(f"Error during model prediction: {e}")
                predicted_label = None

            # 4. Get Disease Name string from the predicted integer label
            if predicted_label is not None:
                try:
                    # Map integer label back to disease name using the 'uniques' array
                    predicted_disease = uniques[predicted_label]
                except IndexError:
                    # Handle cases where the predicted label is out of bounds
                    print(f"Error: Predicted label {predicted_label} is out of bounds for known classes ({len(uniques)}).")
                    predicted_disease = None
            else:
                 # If prediction failed, disease name is None
                 predicted_disease = None

            # 5. Get Treatment Recommendation via OpenAI API (only if disease was predicted)
            treatment_recommendation = "N/A" # Default value
            if predicted_disease: # Check if prediction was successful
                 treatment_recommendation = get_treatment_recommendation_gpt(predicted_disease)

            # 6. Display Results to the user
            print("\n--- Prediction Results ---")
            print(f"Model Used: {model_name}")
            if predicted_disease: # Check if prediction and name lookup succeeded
                print(f"Predicted Disease: {predicted_disease}")
                print(f"Recommended Treatment Info:\n{treatment_recommendation}")
            else:
                # Message if prediction failed or label was invalid
                print("Could not predict disease based on input or prediction error occurred.")
            print("--------------------------")

        # Ask user if they want to perform another prediction
        again = input("\nWould you like to enter another set of symptoms? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            break # Exit the loop if answer is not yes/y

    # Final message when exiting the loop
    print("\nThank you for using the Disease Prediction CLI. Stay healthy!")


# Standard Python entry point
if __name__ == "__main__":
    main()