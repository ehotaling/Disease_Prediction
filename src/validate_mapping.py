from data_utils import load_and_clean_data, normalize_string

"""
Validates that all prognosis values in the cleaned training data
match a canonical disease name in the treatment mapping.

Prints any mismatches after normalization.
"""

# Load datasets (they are already cleaned by your pipeline)
training_df = load_and_clean_data("../data/training_data.csv", "training")
disease_df = load_and_clean_data("../data/Diseases_Symptoms.csv", "diseases_symptoms")

# Normalize all prognosis values in training data
training_prognoses = set(normalize_string(p) for p in training_df["prognosis"].dropna().unique())

# Normalize all canonical disease names from the treatment file
canonical_diseases = set(normalize_string(d) for d in disease_df["Name"].dropna().unique())

# Find mismatches
unmapped_diseases = training_prognoses - canonical_diseases

# Display report
print("=== Mapping Validation Report ===")
print(f"Total unique prognoses in training data: {len(training_prognoses)}")
print(f"Total canonical diseases in treatment mapping: {len(canonical_diseases)}")
print(f"\nUnmapped prognoses ({len(unmapped_diseases)}):")

if unmapped_diseases:
    for disease in sorted(unmapped_diseases):
        print(f" - {disease}")
    print("\n Some prognoses could not be matched. Consider updating your alias map or data.")
else:
    print(" All prognoses from training data are correctly mapped to the treatment dataset!")
