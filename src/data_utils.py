import os
import re
import pandas as pd
from rapidfuzz import process, fuzz

# ============================================================
# File: data_utils.py
# Purpose: Clean and prepare medical symptom training datasets.
#          This includes:
#            - Normalizing strings for matching
#            - Mapping aliases and typos to canonical names
#            - Removing unmatched or unrecognized prognosis entries
# ============================================================

# -----------------------------
# String Normalization Function
# -----------------------------
def normalize_string(s: str) -> str:
    """
    Normalize disease names by:
      - Lowercasing
      - Removing text in parentheses
      - Stripping and collapsing whitespace
    """
    s = s.lower()
    s = re.sub(r'\s*\(.*?\)', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# -----------------------------
# Data Cleaning Utilities
# -----------------------------
def drop_empty_columns(df):
    return df.dropna(axis=1, how='all')

def drop_unnamed_columns(df):
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    return df.drop(columns=unnamed_cols)

# -----------------------------
# Alias Mapping for Common Variants
# -----------------------------
custom_disease_aliases = {
    "(vertigo) paroymsal  positional vertigo": "Labyrinthitis",
    "allergy": "Food Allergy",
    "bronchial asthma": "Asthma",
    "cervical spondylosis": "Spondylosis",
    "chronic cholestasis": "Gestational Cholestasis",
    "common cold": "Influenza (Flu)",
    "dengue": "Dengue Fever",
    "diabetes": "Type 2 Diabetes",
    "dimorphic hemmorhoids(piles)": "Hemorrhoids",
    "fungal infection": "Fungal Infection of the Skin",
    "gerd": "Gastroesophageal Reflux Disease (GERD)",
    "heart attack": "Coronary Atherosclerosis",
    "hepatitis a": "Hepatitis A",
    "hepatitis b": "Hepatitis B",
    "hepatitis c": "Hepatitis C",
    "hepatitis d": "Hepatitis D",
    "hepatitis e": "Hepatitis E",
    "alcoholic hepatitis": "Alcoholic Hepatitis",
    "hypertension": "Hypertensive Heart Disease",
    "jaundice": "Neonatal Jaundice",
    "paralysis (brain hemorrhage)": "Intracranial Hemorrhage"
}

# -----------------------------
# Clean the Diseases_Symptoms Dataset
# -----------------------------
def clean_diseases_symptoms(df):
    df = drop_empty_columns(df)
    df = drop_unnamed_columns(df)

    if 'Name' in df.columns:
        df['Name'] = df['Name'].apply(lambda x: x.strip())

    if 'Symptoms' in df.columns:
        df['Symptoms'] = df['Symptoms'].apply(
            lambda x: ", ".join([s.strip() for s in x.split(",")]) if isinstance(x, str) else x
        )

    if 'Treatments' in df.columns:
        df['Treatments'] = df['Treatments'].apply(
            lambda x: ", ".join([t.strip() for t in x.split(",")]) if isinstance(x, str) else x
        )
        df['Treatments'] = df['Treatments'].fillna("No Treatment Provided")

    return df

# -----------------------------
# Load Canonical Mapping
# -----------------------------
def load_canonical_mapping(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df = clean_diseases_symptoms(df)
    return {row['Name']: normalize_string(row['Name']) for _, row in df.iterrows()}

# -----------------------------
# Clean the Training Dataset
# -----------------------------
def clean_training_data(df):
    df = drop_empty_columns(df)
    df = drop_unnamed_columns(df)

    target = 'prognosis'
    feature_cols = [col for col in df.columns if col != target]

    # Convert all feature columns to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop columns that are now entirely NaN
    df = df.dropna(axis=1, how='all')

    # Load canonical disease names
    disease_csv_path = os.path.join(os.path.dirname(__file__), "../data/Diseases_Symptoms.csv")
    canonical_mapping = load_canonical_mapping(disease_csv_path)
    normalized_to_canonical = {v: k for k, v in canonical_mapping.items()}
    canonical_norms = list(normalized_to_canonical.keys())

    unmatched = []

    # Map each prognosis value
    def map_prognosis(prognosis):
        norm = normalize_string(prognosis)

        # 1. Check custom aliases
        if norm in custom_disease_aliases:
            return custom_disease_aliases[norm]

        # 2. Fuzzy match fallback
        match, score, _ = process.extractOne(norm, canonical_norms, scorer=fuzz.ratio)
        if score >= 80:
            return normalized_to_canonical[match]

        # 3. Log unmatched and return None (to be dropped)
        unmatched.append(prognosis)
        return None

    df[target] = df[target].apply(map_prognosis)
    df = df[df[target].notna()].copy()

    # Print log of dropped prognosis values
    if unmatched:
        print("\nDropped rows due to unmatched diseases:")
        for u in unmatched:
           print(f" - {u}")
    print(f"Total dropped: {len(unmatched)}\n")
    return df

# -----------------------------
# Master Data Loading Function
# -----------------------------
def load_and_clean_data(file_path: str, dataset_type: str):
    df = pd.read_csv(file_path)

    if dataset_type == 'diseases_symptoms':
        return clean_diseases_symptoms(df)
    elif dataset_type == 'training':
        return clean_training_data(df)
    else:
        raise ValueError("dataset_type must be 'diseases_symptoms' or 'training'")
