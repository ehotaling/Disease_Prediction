import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # For isnull check consistency if needed

# Import the primary data loading and cleaning utility function
from data_utils import load_and_clean_data

# ============================================================
# File: clean_data.py
# Purpose: Perform Exploratory Data Analysis (EDA) on the cleaned
#          training dataset to understand its basic characteristics,
#          such as data shape, class distribution, and symptom frequency.
#          Generates visualizations for key distributions.
# ============================================================

# -----------------------------
# Configuration
# -----------------------------
# Define the path to the training data CSV file.
# This assumes the script is run from the 'src/' directory.
training_data_path = "../data/training_data.csv"

# Define constants for plotting top N items (to avoid overly crowded charts)
N_CLASSES_TO_PLOT = 30  # Number of most frequent diseases to visualize
N_SYMPTOMS_TO_PLOT = 50 # Number of most frequent symptoms to visualize

# -----------------------------
# Load Cleaned Data
# -----------------------------
print("Loading training data for EDA...")
try:
    # Use the centralized function from data_utils to load and apply initial cleaning.
    # Expects a DataFrame with feature columns (0/1) and a 'prognosis' column (disease names).
    training_df = load_and_clean_data(training_data_path, 'training')
    print("Training data loaded successfully for EDA.")
except Exception as e:
    # Handle potential errors during data loading (e.g., file not found)
    print(f"FATAL ERROR: Failed to load training data for EDA: {e}")
    # Exit if data cannot be loaded, as EDA cannot proceed
    exit()

# -----------------------------
# Basic Data Inspection
# -----------------------------
print("\n### EDA for Processed Training Data ###")
# Display the dimensions (rows, columns) of the loaded DataFrame
print(f"Shape after initial cleaning: {training_df.shape}")

# Optional: Display data types and check for unexpected missing values
# print("\nData Types:\n", training_df.dtypes)
# missing_counts = training_df.isnull().sum()
# print("\nColumns with Missing Values (if any):\n", missing_counts[missing_counts > 0])

# --------------------------------------------------------
# Analysis of Target Variable (Disease Distribution)
# --------------------------------------------------------
print("\nAnalyzing distribution of 'prognosis' (Disease Classes)...")
# Calculate the frequency of each unique disease name in the 'prognosis' column
prognosis_counts = training_df['prognosis'].value_counts()
# Report the total number of unique diseases found after cleaning/filtering
print(f"Total unique disease classes found: {len(prognosis_counts)}")

# Display the most frequent diseases to understand the head of the distribution
print(f"\nTop {N_CLASSES_TO_PLOT} Most Frequent Diseases:")
# Get the top N most frequent diseases and their counts
top_prognosis_counts = prognosis_counts.head(N_CLASSES_TO_PLOT)
print(top_prognosis_counts)

# --- Plot: Disease Distribution (Top N) ---
print(f"Generating plot for Top {N_CLASSES_TO_PLOT} disease distribution...")
# Create a matplotlib figure and axes for the plot
plt.figure(figsize=(15, 7)) # Set figure size for better readability
# Create a bar chart using the counts of the top N diseases
top_prognosis_counts.plot(kind='bar')
# Set the title and axis labels
plt.title(f"Distribution of Top {N_CLASSES_TO_PLOT} Most Frequent Diseases (Prognosis)")
plt.xlabel("Disease")
plt.ylabel("Frequency (Number of Samples)")
# Rotate x-axis labels for better visibility if names are long
plt.xticks(rotation=75, ha='right')
# Adjust layout to prevent labels overlapping
plt.tight_layout()
# Optionally save the plot to the results directory
# plt.savefig("../results/top_disease_distribution.png")
# Display the plot
plt.show()

# --------------------------------------------------------
# Analysis of Features (Symptom Frequency)
# --------------------------------------------------------
print("\nAnalyzing Symptom Frequency...")
# Identify symptom columns (all columns except the 'prognosis' target column)
symptom_columns = training_df.columns.drop('prognosis')
# Calculate the total number of times each symptom appears (sum of 1s in each column)
# Sort the results to find the most frequent symptoms
symptom_sums = training_df[symptom_columns].sum().sort_values(ascending=False)

# Display the most frequent symptoms
print(f"\nTop {N_SYMPTOMS_TO_PLOT} Most Frequent Symptoms:")
# Get the top N most frequent symptoms and their total counts
top_symptom_sums = symptom_sums.head(N_SYMPTOMS_TO_PLOT)
print(top_symptom_sums)

# --- Plot: Symptom Frequency (Top N) ---
print(f"Generating plot for Top {N_SYMPTOMS_TO_PLOT} symptom frequency...")
# Create a new figure and axes
plt.figure(figsize=(20, 7)) # Use a wider figure for potentially many symptoms
# Create a bar chart using the sums of the top N symptoms
top_symptom_sums.plot(kind='bar')
# Set the title and axis labels
plt.title(f"Frequency of Top {N_SYMPTOMS_TO_PLOT} Symptoms in Training Data")
plt.xlabel("Symptom")
plt.ylabel("Total Occurrences")
# Rotate x-axis labels
plt.xticks(rotation=75, ha='right')
# Adjust layout
plt.tight_layout()
# Optionally save the plot
# plt.savefig("../results/top_symptom_frequency.png")
# Display the plot
plt.show()

# End of script message
print("\nExploratory Data Analysis script finished.")