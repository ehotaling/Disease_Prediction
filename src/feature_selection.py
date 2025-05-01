import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
# Import a simpler estimator for RFE, e.g., DecisionTreeClassifier or LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression # Alternative for RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # For normalizing scores
from data_utils import load_and_clean_data

# ============================================================
# File: feature_selection.py
# Purpose: Identify influential features (symptoms) for predicting
#          disease using multiple methods: Chi-Squared, Mutual Information,
#          Random Forest Importance, and Recursive Feature Elimination (RFE).
#          Generates ranked lists, plots, and a merged ranking.
# ============================================================

# --- Configuration ---
N_FEATURES_TO_PLOT = 15 # Number of top features to plot for each method
N_FEATURES_TO_PRINT = 50 # Number of top features to print for each method
RFE_N_FEATURES = 50 # Number of features RFE should select (can influence ranking)
RESULTS_DIR = "../results"
MODELS_DIR = "../models" # Needed for label mapping
TRAINING_DATA_PATH = "../data/training_data.csv"
DO_SAMPLING_FOR_TESTING = False # Set to False to run on full data (SLOW!)
SAMPLE_SIZE = 20000 # Number of samples if DO_SAMPLING_FOR_TESTING is True

# --- Ensure results directory exists ---
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created directory: {RESULTS_DIR}")

# -----------------------------
# Load and Prepare Data
# -----------------------------
print("Loading dataset...")
df = load_and_clean_data(TRAINING_DATA_PATH, 'training')
print(f"Dataset loaded. Initial shape: {df.shape}")

# --- Optional Sampling for faster testing ---
if DO_SAMPLING_FOR_TESTING:
    print(f"Sampling {SAMPLE_SIZE} rows for testing...")
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Sampled df shape: {df.shape}")

# --- Filtering rare classes (consistent with model_training.py) ---
print("Filtering classes with few samples...")
min_samples_per_class = 3
target_column = 'prognosis'
class_counts = df[target_column].value_counts()
classes_to_keep = class_counts[class_counts >= min_samples_per_class].index
df_filtered = df[df[target_column].isin(classes_to_keep)]
print(f"Shape after filtering rare classes: {df_filtered.shape}")

# --- Define Features (X) and Target (y) ---
X = df_filtered.drop(columns=[target_column])
y = df_filtered[target_column]
print(f"Using {X.shape[1]} features and {y.nunique()} classes after filtering.")

# --- Factorize Target (needed for MI and RFE) ---
print("Factorizing target labels...")
y_encoded, uniques = pd.factorize(y)
# Optionally load the mapping if needed, but factorizing ensures consistency with filtered y
# uniques = np.load(os.path.join(MODELS_DIR, "label_mapping.npy"), allow_pickle=True)

# Store feature names
feature_names = X.columns.tolist()

# --- Split Data (only needed for methods involving model fitting like RF, RFE) ---
# Note: Chi2 and MI can run on the full dataset (X, y_encoded) but using
# X_train is common practice and avoids potential data leakage if results
# were used improperly later. Let's use X, y_encoded directly for Chi2/MI here.
# RFE and RF Importance MUST use training data.
print("Splitting data for RF/RFE training...")
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


# --- DataFrame to store all results ---
all_feature_scores = pd.DataFrame({'Feature': feature_names})

# --------------------------------------------
# Method 1: Chi-Squared Test (SelectKBest)
# --------------------------------------------
print("\nCalculating Chi-Squared scores...")
# Ensure no negative values in X for Chi2 (should be ok with 0/1 data)
if (X < 0).any().any():
     print("Warning: Negative values found in X, Chi2 might be inappropriate.")
     chi2_scores_calculated = np.full(X.shape[1], np.nan) # Handle gracefully
else:
    selector_chi2 = SelectKBest(score_func=chi2, k='all')
    selector_chi2.fit(X, y_encoded) # Fit on full filtered data
    chi2_scores_calculated = selector_chi2.scores_

# Store scores
feature_scores_chi2 = pd.DataFrame({
    'Feature': feature_names,
    'Chi2_Score': chi2_scores_calculated
}).sort_values(by='Chi2_Score', ascending=False).reset_index(drop=True)

print(f"\n--- Top {N_FEATURES_TO_PRINT} Features by Chi-Squared Score ---")
print(feature_scores_chi2.head(N_FEATURES_TO_PRINT))
feature_scores_chi2.to_csv(os.path.join(RESULTS_DIR, "feature_scores_chi2.csv"), index=False)
all_feature_scores = pd.merge(all_feature_scores, feature_scores_chi2, on='Feature', how='left')

# Plot top N
plt.figure(figsize=(12, 6))
top_chi2 = feature_scores_chi2.head(N_FEATURES_TO_PLOT)
plt.bar(top_chi2['Feature'], top_chi2['Chi2_Score'])
plt.title(f"Top {N_FEATURES_TO_PLOT} Features by Chi-Squared Score")
plt.xlabel("Feature")
plt.ylabel("Chi2 Score")
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_features_chi2.png"))
plt.close() # Close plot to avoid displaying immediately

# --------------------------------------------
# Method 2: Mutual Information
# --------------------------------------------
print("\nCalculating Mutual Information scores...")
# MI works well with discrete features and target
mi_scores = mutual_info_classif(X, y_encoded, random_state=42) # Use full filtered data

# Store scores
feature_scores_mi = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False).reset_index(drop=True)

print(f"\n--- Top {N_FEATURES_TO_PRINT} Features by Mutual Information Score ---")
print(feature_scores_mi.head(N_FEATURES_TO_PRINT))
feature_scores_mi.to_csv(os.path.join(RESULTS_DIR, "feature_scores_mi.csv"), index=False)
all_feature_scores = pd.merge(all_feature_scores, feature_scores_mi, on='Feature', how='left')

# Plot top N
plt.figure(figsize=(12, 6))
top_mi = feature_scores_mi.head(N_FEATURES_TO_PLOT)
plt.bar(top_mi['Feature'], top_mi['MI_Score'])
plt.title(f"Top {N_FEATURES_TO_PLOT} Features by Mutual Information Score")
plt.xlabel("Feature")
plt.ylabel("MI Score")
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_features_mi.png"))
plt.close()

# --------------------------------------------
# Method 3: Random Forest Feature Importances
# --------------------------------------------
print("\nCalculating Random Forest feature importances...")
# Train RF on the training split
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("Fitting Random Forest...")
rf.fit(X_train, y_train_encoded)
importances = rf.feature_importances_

# Store scores
feature_scores_rf = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': importances
}).sort_values(by='RF_Importance', ascending=False).reset_index(drop=True)

print(f"\n--- Top {N_FEATURES_TO_PRINT} Features by Random Forest Importance ---")
print(feature_scores_rf.head(N_FEATURES_TO_PRINT))
feature_scores_rf.to_csv(os.path.join(RESULTS_DIR, "feature_scores_rf.csv"), index=False)
all_feature_scores = pd.merge(all_feature_scores, feature_scores_rf, on='Feature', how='left')

# Plot top N
plt.figure(figsize=(12, 6))
top_rf = feature_scores_rf.head(N_FEATURES_TO_PLOT)
plt.bar(top_rf['Feature'], top_rf['RF_Importance'])
plt.title(f"Top {N_FEATURES_TO_PLOT} Features by Random Forest Importance")
plt.xlabel("Feature")
plt.ylabel("Importance (Gini)")
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_features_rf.png"))
plt.close()

# -------------------------------------------------------------
# Method 4: Recursive Feature Elimination (RFE) with Estimator
# -------------------------------------------------------------
print("\nCalculating RFE rankings (can be slow)...")
# Choose an estimator: DecisionTree is usually faster than RF or LR for RFE base
# estimator = LogisticRegression(max_iter=500, solver='liblinear', random_state=42) # Slower
estimator = DecisionTreeClassifier(random_state=42, max_depth=10) # Faster, limit depth
print(f"Using estimator for RFE: {estimator.__class__.__name__}")

# RFE will rank features by recursively removing the least important ones
# Setting step=0.1 removes 10% at each iteration, might be faster
rfe_selector = RFE(estimator=estimator, n_features_to_select=1, step=0.1, verbose=1)
print("Fitting RFE...")
rfe_selector.fit(X_train, y_train_encoded)

# Get rankings (1 is best, higher numbers are worse)
rfe_rankings = rfe_selector.ranking_

# Store rankings
feature_scores_rfe = pd.DataFrame({
    'Feature': feature_names,
    'RFE_Rank': rfe_rankings
}).sort_values(by='RFE_Rank', ascending=True).reset_index(drop=True) # Lower rank is better

print(f"\n--- Top {N_FEATURES_TO_PRINT} Features by RFE Rank ---")
print(feature_scores_rfe.head(N_FEATURES_TO_PRINT))
feature_scores_rfe.to_csv(os.path.join(RESULTS_DIR, "feature_scores_rfe.csv"), index=False)
all_feature_scores = pd.merge(all_feature_scores, feature_scores_rfe, on='Feature', how='left')

# Plot top N (lowest rank number is best)
plt.figure(figsize=(12, 6))
top_rfe = feature_scores_rfe.head(N_FEATURES_TO_PLOT)
# Plot rank itself - lower is better
plt.bar(top_rfe['Feature'], top_rfe['RFE_Rank'])
plt.title(f"Top {N_FEATURES_TO_PLOT} Features by RFE Rank (Lower is Better)")
plt.xlabel("Feature")
plt.ylabel("RFE Rank")
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_features_rfe.png"))
plt.close()

# -------------------------------------
# Create Merged Ranking
# -------------------------------------
print("\nCreating merged feature ranking...")

# Check if all expected columns are present
expected_cols = ['Chi2_Score', 'MI_Score', 'RF_Importance', 'RFE_Rank']
missing_cols = [col for col in expected_cols if col not in all_feature_scores.columns]
if missing_cols:
    print(f"Warning: Missing columns for merging: {missing_cols}")
    # Handle missing columns if necessary, e.g., fill with default bad rank/score

# Fill potential NaNs (e.g., if Chi2 failed) with a value indicating low importance
all_feature_scores.fillna(0, inplace=True) # Fill NaNs in scores with 0
# For RFE Rank, fill NaN with a high number (worse rank)
if 'RFE_Rank' in all_feature_scores.columns:
     max_rank = all_feature_scores['RFE_Rank'].max()
     all_feature_scores['RFE_Rank'].fillna(max_rank + 1, inplace=True)


# --- Normalization ---
# Use MinMaxScaler to scale scores between 0 (worst) and 1 (best)
scaler = MinMaxScaler()

# Scale scores where higher is better (Chi2, MI, RF)
for col in ['Chi2_Score', 'MI_Score', 'RF_Importance']:
     if col in all_feature_scores.columns:
        # Reshape needed for scaler
        scores = all_feature_scores[col].values.reshape(-1, 1)
        all_feature_scores[f'{col}_Norm'] = scaler.fit_transform(scores)

# Invert and scale RFE Rank (lower rank is better, so invert first)
if 'RFE_Rank' in all_feature_scores.columns:
    # Subtract rank from max_rank + 1 to make higher values better
    inverted_rank = (max_rank + 1) - all_feature_scores['RFE_Rank']
    # Scale the inverted rank
    scores = inverted_rank.values.reshape(-1, 1)
    all_feature_scores['RFE_Rank_Norm'] = scaler.fit_transform(scores)

# --- Calculate Combined Score ---
# Simple average of normalized scores
norm_cols = [col for col in all_feature_scores.columns if '_Norm' in col]
if norm_cols:
    all_feature_scores['Combined_Score'] = all_feature_scores[norm_cols].mean(axis=1)
    merged_ranking = all_feature_scores.sort_values(by='Combined_Score', ascending=False).reset_index(drop=True)
    print(f"\n--- Top {N_FEATURES_TO_PRINT} Features by Merged Score ---")
    print(merged_ranking[['Feature', 'Combined_Score'] + norm_cols].head(N_FEATURES_TO_PRINT))
    # Save merged ranking
    merged_ranking.to_csv(os.path.join(RESULTS_DIR, "feature_scores_merged.csv"), index=False)
else:
    print("Could not calculate combined score due to missing normalized columns.")
    merged_ranking = pd.DataFrame() # Empty dataframe


# --- Plot Merged Score ---
if not merged_ranking.empty:
     plt.figure(figsize=(12, 6))
     top_merged = merged_ranking.head(N_FEATURES_TO_PLOT)
     plt.bar(top_merged['Feature'], top_merged['Combined_Score'])
     plt.title(f"Top {N_FEATURES_TO_PLOT} Features by Merged Score")
     plt.xlabel("Feature")
     plt.ylabel("Combined Score (Avg. Normalized)")
     plt.xticks(rotation=60, ha='right')
     plt.tight_layout()
     plt.savefig(os.path.join(RESULTS_DIR, "top_features_merged.png"))
     plt.close()

print("\nFeature selection script finished.")
print(f"Results (scores CSVs and plots PNGs) saved in: {RESULTS_DIR}")

# If plots weren't closed, plt.show() would display them all here
# plt.show()