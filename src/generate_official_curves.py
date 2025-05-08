import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import joblib # For loading scikit-learn models
import torch
import torch.nn as nn
import os

# --- Configuration ---
#  Define base directory for cleaner path construction
BASE_PROJECT_DIR = ".." # Assumes this script is in src/
MODEL_DIR = os.path.join(BASE_PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(BASE_PROJECT_DIR, "results") # Where X_test, y_test_encoded are saved
OFFICIAL_PLOTS_DIR = os.path.join(RESULTS_DIR, "official_evaluation_curves")

# Number of top classes to generate plots for (e.g., for t-SNE consistency or report focus)
N_TOP_CLASSES_TO_PLOT = 10

# --- Create Output Directory if it doesn't exist ---
# Ensures the directory for saving plots is ready
if not os.path.exists(OFFICIAL_PLOTS_DIR):
    os.makedirs(OFFICIAL_PLOTS_DIR)
    print(f"Created directory: {OFFICIAL_PLOTS_DIR}")

# ------------------ PyTorch MLP ------------------

class MLPClassifier(nn.Module):
    """
    Defines the MLP architecture; must match the structure used during training
    in model_training.py for loading the saved weights (state_dict).
    """
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        # Define layers: Input -> Hidden (64 neurons) -> ReLU -> Output
        # This structure is based on the MLP in the provided model_training.py
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        """Defines the forward pass logic of the MLP."""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ------------------ Plotting ------------------
def plot_multiclass_roc_pr(y_true_bin_subset, y_scores_subset, class_names_subset, model_name_str, save_dir):
    """
    Generates and saves ROC and PR curve plots for a given model and subset of classes.
    y_true_bin_subset: Binarized true labels for the selected N classes.
    y_scores_subset: Predicted probabilities/scores for the selected N classes.
    class_names_subset: Names of the selected N classes.
    model_name_str: Name of the model for titles and filenames.
    save_dir: Directory to save the generated plots.
    """
    n_classes_subset = y_true_bin_subset.shape[1]
    # Figure creation
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7)) # Slightly larger for better readability

    # --- ROC curves ---
    # Iterates through each of the N selected classes to plot its One-vs-Rest ROC curve.
    for i in range(n_classes_subset):
        fpr, tpr, _ = roc_curve(y_true_bin_subset[:, i], y_scores_subset[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f'{class_names_subset[i]} (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1) # Diagonal reference line
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curve - {model_name_str}')
    ax_roc.legend(loc="lower right", fontsize='small')
    ax_roc.grid(alpha=0.4)

    # --- PR curves ---
    # Iterates through each of the N selected classes to plot its One-vs-Rest PR curve.
    for i in range(n_classes_subset):
        precision, recall, _ = precision_recall_curve(y_true_bin_subset[:, i], y_scores_subset[:, i])
        ap = average_precision_score(y_true_bin_subset[:, i], y_scores_subset[:, i])
        ax_pr.plot(recall, precision, lw=2, label=f'{class_names_subset[i]} (AP = {ap:.2f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'PR Curve - {model_name_str}')
    ax_pr.legend(loc="lower left", fontsize='small')
    ax_pr.grid(alpha=0.4)

    # Add a super title for the whole figure
    fig.suptitle(f'ROC and PR Curves for Top {len(class_names_subset)} Classes - {model_name_str}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle

    # Construct filename and save the plot
    plot_filename = f"{model_name_str.replace(' ', '_').replace('/', '_')}_top_{len(class_names_subset)}_roc_pr.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close(fig) # Close the figure to free memory

# --- Helper function to get model predictions and prepare for plotting ---
# This function encapsulates loading test data and preparing subsets for plotting.
def get_data_for_plotting():
    """
    Loads the official test set data, full class names, and identifies the top N classes
    for plotting based on their frequency in the test set.
    Returns:
        X_test_all (np.array): Full test feature set.
        y_test_binarized_top_n (np.array): Binarized true labels for top N classes on filtered samples.
        mask_top_n_samples (np.array): Boolean mask for samples belonging to top N classes.
        top_n_class_names (list): List of names for the top N classes.
        class_names_full (np.array): Array of all original class names.
    """
    print("Loading official test data...")
    # Load feature data (X_test)
    # Assumes X_test_data.csv was saved by model_training.py with feature names as columns
    X_test_df = pd.read_csv(os.path.join(RESULTS_DIR, "X_test_data.csv"))
    X_test_all = X_test_df.values

    # Load encoded true labels for the test set (y_test_encoded)
    y_test_encoded_full = np.load(os.path.join(RESULTS_DIR, "y_test_encoded.npy"))

    # Load the mapping from encoded labels to actual disease names ('uniques' array)
    class_names_full = np.load(os.path.join(MODEL_DIR, "label_mapping.npy"), allow_pickle=True)
    print(f"Loaded {len(class_names_full)} total unique class names.")

    print(f"Identifying top {N_TOP_CLASSES_TO_PLOT} classes from the test set for plotting...")
    # Determine unique encoded labels and their counts in the test set
    unique_encoded_labels_in_test, counts_in_test = np.unique(y_test_encoded_full, return_counts=True)

    # Sort these unique labels by their frequency in descending order
    sorted_indices_by_frequency = np.argsort(-counts_in_test)

    # Get the encoded labels of the top N most frequent classes
    top_n_encoded_labels = unique_encoded_labels_in_test[sorted_indices_by_frequency[:N_TOP_CLASSES_TO_PLOT]]

    # Get the actual string names for these top N classes
    top_n_class_names = class_names_full[top_n_encoded_labels]
    print(f"Top {N_TOP_CLASSES_TO_PLOT} classes selected for plotting: {top_n_class_names.tolist()}")

    # Create a boolean mask to identify samples in X_test_all that belong to these top N classes
    mask_top_n_samples = np.isin(y_test_encoded_full, top_n_encoded_labels)

    # Filter the original encoded labels to include only those from the top N classes
    y_test_encoded_for_top_n_samples = y_test_encoded_full[mask_top_n_samples]

    # Binarize these filtered labels. The 'classes' argument ensures correct column mapping for the N classes.
    y_test_binarized_top_n = label_binarize(y_test_encoded_for_top_n_samples, classes=top_n_encoded_labels)

    return X_test_df, X_test_df.values, y_test_binarized_top_n, mask_top_n_samples, top_n_class_names, class_names_full

# ------------------ Main Function ------------------
if __name__ == '__main__':

    # Load data and identify top classes once
    X_test_df_all, X_test_all_numpy, y_test_binarized_top_n, mask_top_n_samples, \
        top_n_class_names, class_names_full = get_data_for_plotting()

    # --- Get indices of top N classes in the full list for column selection ---
    indices_of_top_n_in_full_list = [np.where(class_names_full == cls_name)[0][0] for cls_name in top_n_class_names]

    # --- Process Logistic Regression  ---
    model_name_lr = "Logistic Regression"
    print(f"\nProcessing {model_name_lr}...")
    try:
        lr_model_path = os.path.join(MODEL_DIR, "lr_model.pkl")
        lr_model = joblib.load(lr_model_path)

        # Use X_test_df_all for predict_proba
        lr_all_class_scores = lr_model.predict_proba(X_test_df_all)

        lr_scores_top_n_samples = lr_all_class_scores[mask_top_n_samples]
        lr_scores_for_plotting = lr_scores_top_n_samples[:, indices_of_top_n_in_full_list]
        plot_multiclass_roc_pr(y_test_binarized_top_n, lr_scores_for_plotting,
                               top_n_class_names, model_name_lr, OFFICIAL_PLOTS_DIR)
    except FileNotFoundError:
        print(f"ERROR: {model_name_lr} model not found at {lr_model_path}. Skipping.")
    except Exception as e:
        print(f"ERROR processing {model_name_lr}: {e}")

    # --- Process Random Forest ---
    model_name_rf = "Random Forest"
    print(f"\nProcessing {model_name_rf}...")
    try:
        rf_model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
        rf_model = joblib.load(rf_model_path)

        # MODIFIED: Use X_test_df_all for predict_proba
        rf_all_class_scores = rf_model.predict_proba(X_test_df_all)

        rf_scores_top_n_samples = rf_all_class_scores[mask_top_n_samples]
        rf_scores_for_plotting = rf_scores_top_n_samples[:, indices_of_top_n_in_full_list]
        plot_multiclass_roc_pr(y_test_binarized_top_n, rf_scores_for_plotting,
                               top_n_class_names, model_name_rf, OFFICIAL_PLOTS_DIR)
    except FileNotFoundError:
        print(f"ERROR: {model_name_rf} model not found at {rf_model_path}. Skipping.")
    except Exception as e:
        print(f"ERROR processing {model_name_rf}: {e}")

    # --- Process PyTorch MLP ---
    model_name_mlp = "PyTorch MLP"
    print(f"\nProcessing {model_name_mlp}...")
    try:
        mlp_model_path = os.path.join(MODEL_DIR, "mlp_model.pth")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device} for MLP inference.")
        checkpoint = torch.load(mlp_model_path, map_location=device)
        input_dim = checkpoint['input_dim']
        num_classes_trained = checkpoint['num_classes']
        official_mlp_model = MLPClassifier(input_dim, num_classes_trained).to(device)
        official_mlp_model.load_state_dict(checkpoint['model_state_dict'])
        official_mlp_model.eval()

        # PyTorch MLP still uses the NumPy array (X_test_all_numpy) converted to a tensor
        X_test_all_tensor = torch.tensor(X_test_all_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs_logits = official_mlp_model(X_test_all_tensor)
            mlp_all_class_probabilities = torch.softmax(outputs_logits, dim=1).cpu().numpy()

        mlp_scores_top_n_samples = mlp_all_class_probabilities[mask_top_n_samples]
        mlp_scores_for_plotting = mlp_scores_top_n_samples[:, indices_of_top_n_in_full_list]
        plot_multiclass_roc_pr(y_test_binarized_top_n, mlp_scores_for_plotting,
                               top_n_class_names, model_name_mlp, OFFICIAL_PLOTS_DIR)
    except FileNotFoundError:
        print(f"ERROR: {model_name_mlp} model not found at {mlp_model_path}. Skipping.")
    except Exception as e:
        print(f"ERROR processing {model_name_mlp}: {e}")

    print(f"\nAll official ROC/PR curve plots generation attempt finished.")
    print(f"Plots saved to: {OFFICIAL_PLOTS_DIR}")