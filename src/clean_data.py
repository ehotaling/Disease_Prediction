import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from data_utils import load_and_clean_data

# ============================================================
# File: clean_data.py
# Purpose: Load, clean, and perform exploratory data analysis (EDA)
#          on both the Diseases_Symptoms and training_data datasets.
# ============================================================

# -----------------------------
# Define file paths for the datasets
# -----------------------------
disease_symptoms_path = "../data/Diseases_Symptoms.csv"
training_data_path = "../data/training_data.csv"

# -----------------------------
# Load and clean the datasets using our common utility functions
# -----------------------------
disease_symptoms_df = load_and_clean_data(disease_symptoms_path, 'diseases_symptoms')
training_df = load_and_clean_data(training_data_path, 'training')

# -----------------------------
# EDA for training_data.csv
# -----------------------------
print("### EDA for training_data.csv ###\n")
print("Shape:", training_df.shape)
print("\nData Types:\n", training_df.dtypes)
print("\nMissing Values:\n", training_df.isnull().sum())

# Distribution of the target variable 'prognosis'
print("\nDistribution of 'prognosis':")
prognosis_counts = training_df['prognosis'].value_counts()
print(prognosis_counts)

# Plot the distribution of 'prognosis'
plt.figure(figsize=(10, 6))
prognosis_counts.plot(kind='bar')
plt.title("Distribution of Prognosis (Target Variable)")
plt.xlabel("Prognosis")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Frequency of symptoms across the dataset (excluding the target column)
symptom_columns = training_df.columns.drop('prognosis')
symptom_sums = training_df[symptom_columns].sum().sort_values(ascending=False)
print("\nSymptom Frequency across the dataset:")
print(symptom_sums)

plt.figure(figsize=(20, 6))
symptom_sums.plot(kind='bar')
plt.title("Frequency of Symptoms in Training Data")
plt.xlabel("Symptom")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# EDA for Diseases_Symptoms.csv
# -----------------------------
print("\n### EDA for Diseases_Symptoms.csv ###\n")
print("Shape:", disease_symptoms_df.shape)
print("\nData Types:\n", disease_symptoms_df.dtypes)
print("\nMissing Values:\n", disease_symptoms_df.isnull().sum())

# Count the number of symptoms per disease by splitting the comma-separated list
disease_symptoms_df['Symptom_Count'] = disease_symptoms_df['Symptoms'].apply(lambda x: len(x.split(",")))
print("\nSymptom Count per Disease (first 10 rows):")
print(disease_symptoms_df[['Name', 'Symptom_Count']].head(10))

# Frequency of individual symptoms:
# 1. Split the 'Symptoms' column into individual symptoms.
# 2. Strip extra spaces and convert to lowercase to handle inconsistent casing.
all_symptoms = disease_symptoms_df['Symptoms'].str.split(",").sum()
all_symptoms = [s.strip().lower() for s in all_symptoms]
symptom_frequency = Counter(all_symptoms)
top_symptoms = pd.DataFrame(symptom_frequency.most_common(10), columns=['Symptom', 'Frequency'])
print("\nFrequency of Individual Symptoms in Diseases_Symptoms.csv (Top 10):")
print(top_symptoms)

plt.figure(figsize=(10, 6))
plt.bar(top_symptoms['Symptom'], top_symptoms['Frequency'])
plt.title("Top 10 Symptoms Frequency in Diseases_Symptoms.csv")
plt.xlabel("Symptom")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
