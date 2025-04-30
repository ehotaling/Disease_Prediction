import re

import numpy as np
import pandas as pd

# ============================================================
# File: data_utils.py
# Purpose: Load and perform basic cleaning on the medical symptom training dataset.
# ============================================================

# -----------------------------
# String Normalization Function
# -----------------------------
def normalize_string(s: str) -> str:
    """
    Normalize strings (e.g., disease names) by:
      - Lowercasing
      - Removing text in parentheses (optional, review if needed)
      - Stripping and collapsing whitespace
    """
    if not isinstance(s, str): # Handle potential non-string values
        return s
    s = s.lower()
    # s = re.sub(r'\s*\(.*?\)', '', s) # Keep or remove based on disease name format
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# -----------------------------
# Data Cleaning Utilities
# -----------------------------
def drop_empty_columns(df):
    """Drops columns that contain only NaN values."""
    return df.dropna(axis=1, how='all')

def drop_unnamed_columns(df):
    """Drops columns that start with 'Unnamed'."""
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    return df.drop(columns=unnamed_cols)

# -----------------------------
# Clean the Training Dataset
# -----------------------------
def clean_training_data(df):
    """
    Performs basic cleaning on the training dataset.
    Assumes the first column is the target ('diseases') and others are symptom features (0/1).
    """
    df = drop_unnamed_columns(df)
    df = drop_empty_columns(df)

    # *** Verify the name of the first column containing disease names ***
    target_col_name = df.columns[0] # Assumes first column is target
    print(f"Identified target column: '{target_col_name}'")

    # Get feature column names (all columns except the first one)
    feature_cols = df.columns[1:]
    print(f"Identified {len(feature_cols)} feature columns.")

    # Basic normalization of disease names
    if target_col_name in df.columns:
         print(f"Normalizing target column '{target_col_name}'...")
         df[target_col_name] = df[target_col_name].apply(normalize_string)
         # Checking unique values after normalization
         print("Unique disease names after normalization:", df[target_col_name].nunique())

    # Ensure feature columns are numeric (0 or 1) and handle potential issues
    print("Verifying feature columns...")
    issues_found = False
    for col in feature_cols:
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Check if any NaNs were introduced (means non-numeric data existed)
        if df[col].isnull().any():
             print(f"Warning: Column '{col}' contained non-numeric values. Coerced to NaN.")
             issues_found = True
             # Fill NaNs, e.g., with 0
             df[col] = df[col].fillna(0)

    if issues_found:
         print("Potential issues found in feature columns. Review data or cleaning steps.")

    # Rename the target column consistently to 'prognosis' if needed downstream
    # This keeps compatibility with later scripts expecting 'prognosis'
    if target_col_name != 'prognosis':
        print(f"Renaming target column '{target_col_name}' to 'prognosis'.")
        df = df.rename(columns={target_col_name: 'prognosis'})

    print("Training data cleaning complete.")
    return df

# -----------------------------
# Master Data Loading Function
# -----------------------------
def load_and_clean_data(file_path: str, dataset_type: str):
    """Loads and cleans the specified dataset."""
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

    if dataset_type == 'training':
        return clean_training_data(df)
    else:
        raise ValueError("Invalid dataset_type specified. Expected 'training'.")