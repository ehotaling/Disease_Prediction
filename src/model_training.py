import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from data_utils import load_and_clean_data
import joblib  # For saving the model
import matplotlib.pyplot as plt

# ============================================================
# File: model_training.py
# Purpose: Build and evaluate multiple classification models
#          to predict disease based on symptom data.
#          We use Logistic Regression, Random Forest, and an MLP (via PyTorch).
#          This script also saves the pre-trained Random Forest model to disk.
# ============================================================


# -----------------------------
# Create model directory
# -----------------------------

model_dir = "../models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# -----------------------------
# Load and prepare the dataset
# -----------------------------
training_data_path = "../data/training_data.csv"
df = load_and_clean_data(training_data_path, 'training')

# Define features (X) and target (y)
X = df.drop(columns=['prognosis'])
y = df['prognosis']

# ---------------------------------------------------------------
# Unified factorization of the target labels:
# Factorize the entire target column once so that both the training
# and test sets use the same mapping.
# ---------------------------------------------------------------
y_encoded, uniques = pd.factorize(y)
print("Target classes:", uniques.tolist())

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# Model 1: Logistic Regression (scikit-learn)
# -----------------------------
print("=== Logistic Regression ===")
lr_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
lr_model.fit(X_train, y_train_encoded)  # Train the model
y_pred_lr = lr_model.predict(X_test)  # Predict on the test set
acc_lr = accuracy_score(y_test_encoded, y_pred_lr)



print("Logistic Regression Accuracy: {:.2f}%".format(acc_lr * 100))

# Print detailed classification metrics
print("Classification Report:\n", classification_report(y_test_encoded, y_pred_lr))

# -----------------------------
# Save the Logistic Regression model to disk
# -----------------------------
lr_model_path = os.path.join(model_dir, "lr_model.pkl") # Assuming model_dir is defined as "../models"
joblib.dump(lr_model, lr_model_path)
print(f"\nLogistic Regression model saved to {lr_model_path}")

# -----------------------------
# Model 2: Random Forest Classifier (scikit-learn)
# -----------------------------
print("\n=== Random Forest Classifier ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_encoded)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test_encoded, y_pred_rf)
print("Random Forest Accuracy: {:.2f}%".format(acc_rf * 100))

# Print detailed classification metrics
print("Classification Report:\n", classification_report(y_test_encoded, y_pred_rf))

# -----------------------------
# Save the Random Forest model to disk
# -----------------------------

model_path = os.path.join(model_dir, "rf_model.pkl")
joblib.dump(rf_model, model_path)
print(f"Random Forest model saved to {model_path}")

# -----------------------------
# Model 3: MLP Classifier using PyTorch
# -----------------------------
print("\n=== PyTorch MLP Classifier ===")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert training and testing data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# Create a DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define a simple MLP model with one hidden layer
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Hidden layer with 64 neurons
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(64, num_classes)  # Output layer for class logits

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_dim = X_train.shape[1]
num_classes = len(uniques)
mlp_model = MLPClassifier(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    mlp_model.train()  # Set model to training mode
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = mlp_model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate the MLP model on the test set
mlp_model.eval()  # Set model to evaluation mode
with torch.no_grad():
    outputs = mlp_model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)  # Get predicted class labels
    acc_torch = (predicted == y_test_tensor).float().mean().item()
    print("PyTorch MLP Accuracy: {:.2f}%".format(acc_torch * 100))

    # Compute and print the detailed classification report for the PyTorch model
    predicted_np = predicted.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    print("Classification Report:\n", classification_report(y_test_np, predicted_np))

# -----------------------------
# Save the PyTorch MLP model to disk
# -----------------------------
mlp_model_path = os.path.join(model_dir, "mlp_model.pth")
torch.save(mlp_model.state_dict(), mlp_model_path)
print(f"PyTorch MLP model state_dict saved to {mlp_model_path}")
# -------------------------------------
# Compute additional performance metrics for all models
# -------------------------------------
# Logistic Regression metrics
precision_lr = precision_score(y_test_encoded, y_pred_lr, average='macro')
recall_lr = recall_score(y_test_encoded, y_pred_lr, average='macro')
f1_lr = f1_score(y_test_encoded, y_pred_lr, average='macro')

# Random Forest metrics
precision_rf = precision_score(y_test_encoded, y_pred_rf, average='macro')
recall_rf = recall_score(y_test_encoded, y_pred_rf, average='macro')
f1_rf = f1_score(y_test_encoded, y_pred_rf, average='macro')

# PyTorch MLP metrics (using numpy arrays from the PyTorch evaluation)
precision_torch = precision_score(y_test_np, predicted_np, average='macro')
recall_torch = recall_score(y_test_np, predicted_np, average='macro')
f1_torch = f1_score(y_test_np, predicted_np, average='macro')

# Create lists for each metric (converted to percentages)
model_names = ["Logistic Regression", "Random Forest", "PyTorch MLP"]
accuracies = [acc_lr * 100, acc_rf * 100, acc_torch * 100]
precisions = [precision_lr * 100, precision_rf * 100, precision_torch * 100]
recalls = [recall_lr * 100, recall_rf * 100, recall_torch * 100]
f1_scores = [f1_lr * 100, f1_rf * 100, f1_torch * 100]

# -------------------------------------
# Plot individual bar charts for model accuracy
# -------------------------------------
for model_name, acc in zip(model_names, accuracies):
    plt.figure()
    plt.bar(model_name, acc)
    plt.title(f"{model_name} Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y')

# Combined comparison chart for accuracy
plt.figure()
plt.bar(model_names, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis='y')

# Combined comparison chart for precision
plt.figure()
plt.bar(model_names, precisions)
plt.title("Model Precision Comparison")
plt.ylabel("Precision (%)")
plt.ylim(0, 100)
plt.grid(axis='y')

# Combined comparison chart for recall
plt.figure()
plt.bar(model_names, recalls)
plt.title("Model Recall Comparison")
plt.ylabel("Recall (%)")
plt.ylim(0, 100)
plt.grid(axis='y')

# Combined comparison chart for F1 score
plt.figure()
plt.bar(model_names, f1_scores)
plt.title("Model F1 Score Comparison")
plt.ylabel("F1 Score (%)")
plt.ylim(0, 100)
plt.grid(axis='y')

plt.show()
