import pandas as pd


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any columns that are completely empty (all values are NaN).

    This function is useful because sometimes data files include empty columns
    that are not needed for analysis.
    """
    return df.dropna(axis=1, how='all')


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns whose names start with 'Unnamed'.

    Many CSV exports include columns like 'Unnamed: 133' which usually contain no useful data.
    """
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    return df.drop(columns=unnamed_cols)


def clean_diseases_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Diseases_Symptoms dataset.

    Steps:
      1. Drop empty columns and any unnamed columns.
      2. Clean the 'Symptoms' column:
         - Split the string by commas and strip extra spaces.
      3. Clean the 'Treatments' column:
         - Similarly split and strip text.
         - Fill in missing values with "No Treatment Provided".
    """
    # Remove unnecessary columns
    df = drop_empty_columns(df)
    df = drop_unnamed_columns(df)

    # Clean the 'Symptoms' column if it exists
    if 'Symptoms' in df.columns:
        df['Symptoms'] = df['Symptoms'].apply(
            lambda x: ", ".join([sym.strip() for sym in x.split(",")]) if isinstance(x, str) else x
        )

    # Clean the 'Treatments' column if it exists
    if 'Treatments' in df.columns:
        df['Treatments'] = df['Treatments'].apply(
            lambda x: ", ".join([t.strip() for t in x.split(",")]) if isinstance(x, str) else x
        )
        df['Treatments'] = df['Treatments'].fillna("No Treatment Provided")

    return df


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the training_data dataset.

    Steps:
      1. Drop empty columns and any columns with names starting with 'Unnamed'.
      2. For all feature columns (all columns except the target 'prognosis'):
         - Attempt to convert them to numeric if they are not already.
           (This makes the cleaning process agnostic to future changes in the data.)
    """
    # Remove unnecessary columns
    df = drop_empty_columns(df)
    df = drop_unnamed_columns(df)

    # Define the target column
    target = 'prognosis'
    # Process all columns except the target
    feature_cols = [col for col in df.columns if col != target]
    for col in feature_cols:
        # Convert column to numeric if it is not already (non-numeric values become NaN)
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any feature columns that may have become entirely NaN after conversion
    df = df.dropna(axis=1, how='all')
    return df


def load_and_clean_data(file_path: str, dataset_type: str) -> pd.DataFrame:
    """
    Load a CSV file and clean it based on its type.

    Parameters:
      - file_path: Path to the CSV file.
      - dataset_type: A string indicating the type of dataset. Acceptable values are:
                      'diseases_symptoms' or 'training'

    Returns:
      - A cleaned pandas DataFrame.

    Raises:
      - ValueError if an unknown dataset_type is provided.
    """
    df = pd.read_csv(file_path)
    if dataset_type == 'diseases_symptoms':
        df = clean_diseases_symptoms(df)
    elif dataset_type == 'training':
        df = clean_training_data(df)
    else:
        raise ValueError("Unknown dataset type. Please use 'diseases_symptoms' or 'training'.")
    return df
