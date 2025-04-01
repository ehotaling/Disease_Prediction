import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_utils import load_and_clean_data

# ============================================================
# File: feature_selection.py
# Purpose: Identify the most influential features (symptoms)
#          for predicting the disease ('prognosis') using
#          both a statistical test (chi-squared) and a Random Forest.
# ============================================================

# -----------------------------
# Load and clean the training dataset
# -----------------------------
training_data_path = "../data/training_data.csv"
df = load_and_clean_data(training_data_path, 'training')

# -----------------------------
# Prepare features and target variables
# -----------------------------
# X contains all the symptom features; y is the target (disease prognosis)
X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Split data into training and test sets to evaluate feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Feature Selection using Chi-Squared Test (SelectKBest)
# -----------------------------
# Chi-squared test is used to determine the relationship between categorical features and the target.
selector = SelectKBest(score_func=chi2, k='all')
selector.fit(X_train, y_train)

# Get the chi-squared scores for each feature and sort them
chi2_scores = selector.scores_
features = X_train.columns
feature_scores = pd.DataFrame({'Feature': features, 'Chi2_Score': chi2_scores})
feature_scores = feature_scores.sort_values(by='Chi2_Score', ascending=False)
print("Top 50 Features by Chi-Squared Score:")
print(feature_scores.head(50))

# Plot the top 10 features based on chi-squared scores
plt.figure(figsize=(12, 6))
plt.bar(feature_scores['Feature'].head(10), feature_scores['Chi2_Score'].head(10))
plt.title("Top 10 Features by Chi-Squared Score")
plt.xlabel("Feature")
plt.ylabel("Chi2 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Feature Selection using Random Forest Importances
# -----------------------------
# Random Forest provides an importance score for each feature based on its contribution to reducing error.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# Create a DataFrame for the feature importances and sort them
rf_feature_importances = pd.DataFrame({'Feature': features, 'RF_Importance': importances})
rf_feature_importances = rf_feature_importances.sort_values(by='RF_Importance', ascending=False)
print("Top 10 Features by Random Forest Importance:")
print(rf_feature_importances.head(10))

# Plot the top 10 features based on Random Forest importance
plt.figure(figsize=(12, 6))
plt.bar(rf_feature_importances['Feature'].head(10), rf_feature_importances['RF_Importance'].head(10))
plt.title("Top 10 Features by Random Forest Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
