import os
import pandas as pd
import numpy as np
# Set environment variable BEFORE importing torch or other libraries that might initialize CUDA
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from data_utils import load_and_clean_data
import joblib  # For saving the model
import matplotlib.pyplot as plt
import torch # Ensure torch is imported early if used globally
import torch.nn as nn
import torch.optim as optim
import copy
import logging
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


# ============================================================
# File: model_training.py
# Purpose: Build and evaluate multiple classification models
#          to predict disease based on symptom data.
#          We use Logistic Regression, Random Forest, and an MLP (via PyTorch).
#          This script also saves the pre-trained models to disk.
# ============================================================


# -----------------------------
# Create model directory
# -----------------------------

model_dir = "../models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")
else:
    print(f"Directory already exists: {model_dir}")

# -----------------------------
# Create results directory (if needed)
# -----------------------------

results_dir = "../results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")
else:
    print(f"Directory already exists: {results_dir}")

# --- Configure Logging (File Only for Important Info) ---
important_log_file_path = os.path.join(results_dir, "important_training_summary.log")

# Create a specific logger instance
file_logger = logging.getLogger('summary_logger')
file_logger.setLevel(logging.INFO) # Set the minimum level for this logger

# Prevent propagation to root logger (if it has handlers like console)
file_logger.propagate = False

# Remove existing handlers for this specific logger (if any)
for handler in file_logger.handlers[:]:
    file_logger.removeHandler(handler)

# Create File handler ONLY
file_handler = logging.FileHandler(important_log_file_path, mode='w') # 'w' to overwrite
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the file handler
# Simple format for the summary log
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
file_logger.addHandler(file_handler)

# Log the start and the log file path itself
file_logger.info("Starting model training script run.")
file_logger.info("Saving important summary log to: %s", important_log_file_path)
print(f"Saving important summary log to: {important_log_file_path}") # Also print this info
# --- End of Logging Setup ---


# -----------------------------
# Load and prepare the dataset
# -----------------------------
training_data_path = "../data/training_data.csv"
print("Loading dataset...")
df = load_and_clean_data(training_data_path, 'training')
print("Dataset loaded.")
file_logger.info("Dataset loaded from: %s", training_data_path)

print(f"Filtering classes with few samples...")
min_samples_per_class = 3 # Or 2, or 5 - your choice
target_column = 'prognosis' # Use 'prognosis' as it's renamed by data_utils

class_counts = df[target_column].value_counts()

# Identify classes to REMOVE
classes_to_remove = class_counts[class_counts < min_samples_per_class].index
print(f"\nClasses with < {min_samples_per_class} samples (to be removed):")
if not classes_to_remove.empty:
    # Print the list of removed classes
    print(classes_to_remove.tolist())
else:
    print("None")

# Identify classes to KEEP
classes_to_keep = class_counts[class_counts >= min_samples_per_class].index

original_rows = df.shape[0]
original_classes = df[target_column].nunique()

# Perform the filtering
df_filtered = df[df[target_column].isin(classes_to_keep)]

filtered_rows = df_filtered.shape[0]
filtered_classes = df_filtered[target_column].nunique()

print(f"\nRemoved {original_rows - filtered_rows} rows belonging to {original_classes - filtered_classes} classes with < {min_samples_per_class} samples.")
print(f"Shape after filtering: {df_filtered.shape}")
file_logger.info("Data shape after filtering classes with < %d samples: %s", min_samples_per_class, df_filtered.shape) # Add summary to file log
file_logger.info("Removed %d rows and %d classes.", original_rows - filtered_rows, original_classes - filtered_classes) # Add summary to file log

# Now use df_filtered for defining X and y
X = df_filtered.drop(columns=[target_column]) # Use variable for target column name
y = df_filtered[target_column]
# --- End of filtering block ---

# Factorize the filtered target labels
print("\nFactorizing filtered target labels...")
y_encoded, uniques = pd.factorize(y)
print(f"Number of unique classes after filtering: {len(uniques)}")
file_logger.info("Number of unique classes: %d", len(uniques)) # Add for file log
# Save the label mapping for future prediction use
label_map_path = os.path.join(model_dir, "label_mapping.npy")
np.save(label_map_path, uniques)
print(f"Label mapping saved to {label_map_path}")
file_logger.info("Label mapping saved to: %s", label_map_path) # Add for file log

# Split the filtered data
print("Splitting data into train/val/test sets...")

# First, split off the test set (20%)
# X is a DataFrame, so X_test_data will also be a DataFrame
X_temp, X_test, y_temp, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Then split the remaining 80% into train and validation sets (64% train, 16% val)
X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"Training set size:     {X_train.shape[0]} samples")
print(f"Validation set size:   {X_val.shape[0]} samples")
print(f"Test set size:         {X_test.shape[0]} samples")
file_logger.info("Data split sizes - Train: %d, Validation: %d, Test: %d", X_train.shape[0], X_val.shape[0], X_test.shape[0])

# Save the test set for generate_curves.py
print("\nSaving test set data for external curve generation...")
try:
    # X_test_data is already a pandas DataFrame with column names from X
    x_test_save_path = os.path.join(results_dir, "X_test_data.csv")
    y_test_save_path = os.path.join(results_dir, "y_test_encoded.npy")

    X_test.to_csv(x_test_save_path, index=False) # Directly save the DataFrame
    np.save(y_test_save_path, y_test_encoded)

    print(f"X_test data saved to: {x_test_save_path}")
    print(f"y_test_encoded data saved to: {y_test_save_path}")
    file_logger.info("Test set (X_test_data.csv, y_test_encoded.npy) saved to: %s", results_dir)
except Exception as e:
    print(f"Error saving test set data: {e}")
    file_logger.error("Error saving test set data: %s", e)

# -----------------------------
# Model 1: Logistic Regression (scikit-learn)
# -----------------------------
print("\n=== Training Logistic Regression ===")
lr_model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1) # Use available cores
lr_model.fit(X_train, y_train_encoded)
print("Logistic Regression training complete.")
print("Evaluating Logistic Regression...")
y_pred_lr = lr_model.predict(X_test)
acc_lr = accuracy_score(y_test_encoded, y_pred_lr)
print("Logistic Regression Accuracy: {:.2f}%".format(acc_lr * 100))
file_logger.info("Logistic Regression - Test Accuracy: %.2f%%", acc_lr * 100) # Add final acc to file log
# print("Classification Report:\n", classification_report(y_test_encoded, y_pred_lr, zero_division=0))

# Save the Logistic Regression model
print("Saving Logistic Regression model...")
lr_model_path = os.path.join(model_dir, "lr_model.pkl")
joblib.dump(lr_model, lr_model_path)
print(f"Logistic Regression model saved to {lr_model_path}")
file_logger.info("Logistic Regression model saved to: %s", lr_model_path) # Add for file log

# -----------------------------
# Model 2: Random Forest Classifier (scikit-learn)
# -----------------------------
print("\n=== Training Random Forest Classifier ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Use available cores
rf_model.fit(X_train, y_train_encoded)
print("Random Forest training complete.")
print("Evaluating Random Forest...")
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test_encoded, y_pred_rf)
print("Random Forest Accuracy: {:.2f}%".format(acc_rf * 100))
file_logger.info("Random Forest - Test Accuracy: %.2f%%", acc_rf * 100) # Add final acc to file log
# print("Classification Report:\n", classification_report(y_test_encoded, y_pred_rf, zero_division=0))

# Save the Random Forest model
print("Saving Random Forest model...")
rf_model_path = os.path.join(model_dir, "rf_model.pkl")
joblib.dump(rf_model, rf_model_path)
print(f"Random Forest model saved to {rf_model_path}")
file_logger.info("Random Forest model saved to: %s", rf_model_path) # Add for file log

# -----------------------------
# Model 3: MLP Classifier using PyTorch
# -----------------------------
print("\n=== Training PyTorch MLP Classifier ===")

# Convert training and testing data to PyTorch tensors
print("Converting data to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
# Also convert validation data to tensors
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=1024)  # Big batch for eval

print("Tensor conversion complete.")

# Create a DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# Consider increasing batch_size for larger dataset if memory allows
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Increased batch_size


# A simple MLP model with one hidden layer
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_dim = X_train.shape[1]
num_classes = len(uniques)
# Check for available device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
file_logger.info("MLP using device: %s", device) # Add for file log
mlp_model = MLPClassifier(input_dim, num_classes).to(device) # Move model to device

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

num_epochs = 15 # Can change this
best_val_acc = 0.0
epochs_without_improvement = 0
patience = 3  # Stop training if no improvement after 3 epochs
print(f"Starting MLP training for {num_epochs} epochs...")
file_logger.info("MLP training started (Max Epochs: %d, Patience: %d)", num_epochs, patience) # Add for file log
for epoch in range(num_epochs):
    mlp_model.train()
    epoch_loss = 0
    processed_samples = 0
    for i, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = mlp_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        processed_samples += len(batch_X)

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {avg_epoch_loss:.4f}")

    # --- Validation Accuracy ---
    mlp_model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for val_X_batch, val_y_batch in val_loader:
            val_X_batch = val_X_batch.to(device)
            val_y_batch = val_y_batch.to(device)
            val_outputs = mlp_model(val_X_batch)
            _, val_predicted = torch.max(val_outputs, 1)
            val_preds.extend(val_predicted.cpu().numpy())
            val_true.extend(val_y_batch.cpu().numpy())

    val_acc = accuracy_score(val_true, val_preds)
    print(f"  Validation Accuracy: {val_acc:.2%}")

    # --- Early Stopping Check ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        # Save best model during training
        best_model_state = copy.deepcopy(mlp_model.state_dict())
        print("  New best model found. Saving checkpoint.")
    else:
        epochs_without_improvement += 1
        print(f"  No improvement. Patience: {epochs_without_improvement}/{patience}")
        if epochs_without_improvement >= patience:
            print("  Early stopping triggered.")
            file_logger.info("MLP early stopping triggered at epoch %d. Best Validation Acc: %.2f%%", epoch + 1,
                             best_val_acc * 100)  # Add for file log
            break

# Load best model before final evaluation on test set
mlp_model.load_state_dict(best_model_state)
print("Best model loaded for final evaluation.")
file_logger.info("MLP training finished. Best Validation Acc: %.2f%%", best_val_acc * 100) # Add summary to file log


print("MLP training complete.")
print("Evaluating MLP...")
# Evaluate the MLP model on the test set
mlp_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    # Process test set potentially in batches if it's large
    test_dataset_eval = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader_eval = DataLoader(test_dataset_eval, batch_size=1024) # Use larger batch for eval

    for batch_X_test, batch_y_test in test_loader_eval:
         batch_X_test = batch_X_test.to(device)
         outputs = mlp_model(batch_X_test)
         _, predicted = torch.max(outputs.data, 1)
         all_preds.extend(predicted.cpu().numpy()) # Collect predictions
         all_labels.extend(batch_y_test.cpu().numpy()) # Collect true labels

predicted_np = np.array(all_preds)
y_test_np = np.array(all_labels) # Use collected true labels

acc_torch = accuracy_score(y_test_np, predicted_np)
print("PyTorch MLP Accuracy: {:.2f}%".format(acc_torch * 100))
file_logger.info("PyTorch MLP - Test Accuracy: %.2f%%", acc_torch * 100) # Add final acc to file log
# print("Classification Report:\n", classification_report(y_test_np, predicted_np, zero_division=0)) # Commented out

# Save the PyTorch MLP model state dictionary
print("Saving PyTorch MLP model...")
mlp_model_path = os.path.join(model_dir, "mlp_model.pth")
torch.save({
    'model_state_dict': mlp_model.state_dict(),
    'input_dim': input_dim,
    'num_classes': num_classes
}, mlp_model_path)

print(f"PyTorch MLP model state_dict saved to {mlp_model_path}")
file_logger.info("PyTorch MLP model saved to: %s", mlp_model_path) # Add for file log

# -------------------------------------
# Compute final performance metrics for all models
# -------------------------------------
print("\nComputing final performance metrics (macro averages)...")
# Logistic Regression metrics
precision_lr = precision_score(y_test_encoded, y_pred_lr, average='macro', zero_division=0)
recall_lr = recall_score(y_test_encoded, y_pred_lr, average='macro', zero_division=0)
f1_lr = f1_score(y_test_encoded, y_pred_lr, average='macro', zero_division=0)

# Random Forest metrics
precision_rf = precision_score(y_test_encoded, y_pred_rf, average='macro', zero_division=0)
recall_rf = recall_score(y_test_encoded, y_pred_rf, average='macro', zero_division=0)
f1_rf = f1_score(y_test_encoded, y_pred_rf, average='macro', zero_division=0)

# PyTorch MLP metrics
precision_torch = precision_score(y_test_np, predicted_np, average='macro', zero_division=0)
recall_torch = recall_score(y_test_np, predicted_np, average='macro', zero_division=0)
f1_torch = f1_score(y_test_np, predicted_np, average='macro', zero_division=0)

file_logger.info("--- Final Performance Metrics (Macro Avg) ---") # Add header to file log
file_logger.info("Logistic Regression - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%", precision_lr*100, recall_lr*100, f1_lr*100)
file_logger.info("Random Forest - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%", precision_rf*100, recall_rf*100, f1_rf*100)
file_logger.info("PyTorch MLP - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%", precision_torch*100, recall_torch*100, f1_torch*100)

# Create lists for each metric (converted to percentages)
model_names = ["Logistic Regression", "Random Forest", "PyTorch MLP"]
accuracies = [acc_lr * 100, acc_rf * 100, acc_torch * 100]
precisions = [precision_lr * 100, precision_rf * 100, precision_torch * 100]
recalls = [recall_lr * 100, recall_rf * 100, recall_torch * 100]
f1_scores = [f1_lr * 100, f1_rf * 100, f1_torch * 100]
print("Metrics computed.")

# -------------------------------------
# Create and Print Comparison Table
# -------------------------------------
print("\n--- Model Performance Comparison ---")
# Create a dictionary for the scores
# Uses rounded percentages for display in table
scores_dict = {
    'Accuracy (%)': [round(acc, 2) for acc in accuracies],
    'Precision (Macro, %)': [round(p, 2) for p in precisions],
    'Recall (Macro, %)': [round(r, 2) for r in recalls],
    'F1-Score (Macro, %)': [round(f1, 2) for f1 in f1_scores]
}

# Create DataFrame
comparison_df = pd.DataFrame(scores_dict, index=model_names)

# Print the DataFrame
print(comparison_df.to_markdown(numalign="left", stralign="left")) # Print as Markdown table

# Create a results directory if it doesn't exist already
results_dir = "../results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")


# Now save the table
comparison_df.to_csv(os.path.join(results_dir, "model_comparison.csv"))
print(f"\nComparison table saved to {os.path.join(results_dir, 'model_comparison.csv')}")
file_logger.info("Model comparison table saved to: %s", os.path.join(results_dir, "model_comparison.csv")) # Add for file log


# -------------------------------------
# Create Matplotlib Table Visualization
# -------------------------------------
print("\nGenerating visual comparison table...")

# Data needs to be transposed from comparison_df
# And formatted as strings with '%'
df_display = comparison_df.T # Transpose: Metrics as rows, Models as columns
for col in df_display.columns:
    df_display[col] = df_display[col].map(lambda x: f"{x:.2f}%")


# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 3)) # Adjust figsize as needed

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create the table
tab = plt.table(cellText=df_display.values,
                rowLabels=df_display.index,
                colLabels=df_display.columns,
                cellLoc = 'center',
                loc='center')

# Adjust layout and title
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1.2, 1.2) # Adjust scale as needed
plt.title('Model Performance Comparison', fontsize=14, y=1.08) # Adjust title position
plt.tight_layout(pad=2.0)

# Optionally save the table figure
# Ensure results_dir is defined earlier
# results_dir = "../results"
# if not os.path.exists(results_dir): os.makedirs(results_dir)
plt.savefig(os.path.join(results_dir, "model_comparison_table.png"), bbox_inches='tight', dpi=150)
print(f"Comparison table plot saved to {os.path.join(results_dir, 'model_comparison_table.png')}")
file_logger.info("Model comparison table plot saved to: %s", os.path.join(results_dir, "model_comparison_table.png")) # Add for file log


# -------------------------------------
# Plotting (Combined Bar Charts)
# -------------------------------------
print("\nGenerating comparison bar charts...")

# Combined comparison chart for accuracy
plt.figure(figsize=(8, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, max(accuracies) + 10 if accuracies else 105) # Adjust ylim based on data
plt.grid(axis='y', linestyle='--')
# Add text labels
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"))

# Combined comparison chart for precision
plt.figure(figsize=(8, 6))
plt.bar(model_names, precisions, color=['blue', 'green', 'red'])
plt.title("Model Precision Comparison (Macro Avg)")
plt.ylabel("Precision (%)")
plt.ylim(0, max(precisions) + 10 if precisions else 105)
plt.grid(axis='y', linestyle='--')
for i, p in enumerate(precisions):
    plt.text(i, p + 1, f'{p:.2f}%', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "precision_comparison.png"))

# Combined comparison chart for recall
plt.figure(figsize=(8, 6))
plt.bar(model_names, recalls, color=['blue', 'green', 'red'])
plt.title("Model Recall Comparison (Macro Avg)")
plt.ylabel("Recall (%)")
plt.ylim(0, max(recalls) + 10 if recalls else 105)
plt.grid(axis='y', linestyle='--')
for i, r in enumerate(recalls):
    plt.text(i, r + 1, f'{r:.2f}%', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "recall_comparison.png"))

# Combined comparison chart for F1 score
plt.figure(figsize=(8, 6))
plt.bar(model_names, f1_scores, color=['blue', 'green', 'red'])
plt.title("Model F1 Score Comparison (Macro Avg)")
plt.ylabel("F1 Score (%)")
plt.ylim(0, max(f1_scores) + 10 if f1_scores else 105)
plt.grid(axis='y', linestyle='--')
for i, f1 in enumerate(f1_scores):
    plt.text(i, f1 + 1, f'{f1:.2f}%', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "f1_score_comparison.png"))

file_logger.info("Comparison plots (Accuracy, Precision, Recall, F1) saved in %s", results_dir) # Add summary to file log

print("Displaying plots...")
plt.show()
print("Model training script finished.")
file_logger.info("Model training script finished.") # Add for file log