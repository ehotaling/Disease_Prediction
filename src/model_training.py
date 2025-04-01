import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_utils import load_and_clean_data

# ============================================================
# File: model_training.py
# Purpose: Build and evaluate multiple classification models
#          to predict disease based on symptom data.
#          We use Logistic Regression, Random Forest, and an MLP (via PyTorch).
# ============================================================

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
# We factorize the entire target column once so that both the
# training and test sets use the same mapping.
# ---------------------------------------------------------------
y_encoded, uniques = pd.factorize(y)
print("Target classes:", uniques)

# Split the data into training and testing sets using the unified encoding
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# Model 1: Logistic Regression (scikit-learn)
# -----------------------------
# Logistic Regression is a linear model suitable for multi-class classification.
print("=== Logistic Regression ===")
lr_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
lr_model.fit(X_train, y_train_encoded)  # Train the model
y_pred_lr = lr_model.predict(X_test)  # Predict on the test set

# Calculate accuracy and display a detailed classification report
acc_lr = accuracy_score(y_test_encoded, y_pred_lr)
print("Logistic Regression Accuracy: {:.2f}%".format(acc_lr * 100))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred_lr))

# -----------------------------
# Model 2: Random Forest Classifier (scikit-learn)
# -----------------------------
# Random Forest is an ensemble method using multiple decision trees.
print("\n=== Random Forest Classifier ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_encoded)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test_encoded, y_pred_rf)
print("Random Forest Accuracy: {:.2f}%".format(acc_rf * 100))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred_rf))

# -----------------------------
# Model 3: MLP Classifier using PyTorch
# -----------------------------
# We implement a simple Multi-Layer Perceptron (MLP) using PyTorch.
print("\n=== PyTorch MLP Classifier ===")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert training and testing data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# Use the unified encoding for target labels
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# Create a DataLoader for batch processing of training data
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


# Determine the number of features and classes
input_dim = X_train.shape[1]
num_classes = len(uniques)
model = MLPClassifier(input_dim, num_classes)

# Define loss function (CrossEntropyLoss) and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for the MLP model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0  # Initialize epoch loss
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(batch_X)  # Forward pass: compute predictions
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters
        epoch_loss += loss.item()  # Accumulate loss
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate the MLP model on the test set
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)  # Get predicted class labels
    acc_torch = (predicted == y_test_tensor).float().mean().item()
    print("PyTorch MLP Accuracy: {:.2f}%".format(acc_torch * 100))
