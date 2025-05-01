import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import data_utils # Import the utility functions from your data_utils.py file
import time # To measure execution time

# --- Configuration ---
TRAINING_DATA_PATH = "../data/training_data.csv"
RESULTS_DIR = "../results" # Directory to save the plot
DO_SAMPLING = True # Set to True to run on a sample, False for full data (SLOW!)
SAMPLE_SIZE = 5000 # Number of samples if DO_SAMPLING is True
MIN_SAMPLES_PER_CLASS = 3 # Minimum samples per class to keep (consistent with model_training.py)
N_TOP_CLASSES = 20 # Number of top classes to visualize
TSNE_PERPLEXITY = 30 # Typical value, adjust as needed
TSNE_N_ITER = 1000 # Number of iterations for optimization
TSNE_LEARNING_RATE = 200 # Typical value, adjust as needed


# --- Ensure results directory exists ---
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created directory: {RESULTS_DIR}")

# --- Load and Prepare Data ---
print("Loading and cleaning dataset...")
try:
    # Use the function from data_utils.py
    df = data_utils.load_and_clean_data(TRAINING_DATA_PATH, 'training') #
    print(f"Dataset loaded. Initial shape: {df.shape}")

    target_column = 'prognosis' # Defined in data_utils cleaning

    # --- Filtering rare classes (consistent with model_training.py) ---
    print(f"Filtering classes with < {MIN_SAMPLES_PER_CLASS} samples...") #
    class_counts = df[target_column].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
    df_filtered = df[df[target_column].isin(classes_to_keep)].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Shape after filtering rare classes: {df_filtered.shape}")
    print(f"Number of unique classes after filtering: {df_filtered[target_column].nunique()}")

    # --- Identify and Filter for Top N Classes ---
    print(f"\nIdentifying top {N_TOP_CLASSES} most frequent classes...")
    top_n_classes = df_filtered[target_column].value_counts().nlargest(N_TOP_CLASSES).index.tolist()
    print(f"Top {N_TOP_CLASSES} classes: {top_n_classes}")

    # Filter the DataFrame to keep only rows belonging to the top N classes
    df_top_n = df_filtered[df_filtered[target_column].isin(top_n_classes)].copy() # Use .copy()
    print(f"Shape after filtering for top {N_TOP_CLASSES} classes: {df_top_n.shape}")

    # --- Optional Sampling (Applied to the Top N subset) ---
    if DO_SAMPLING:
        if SAMPLE_SIZE >= df_top_n.shape[0]:
             print(f"Sample size ({SAMPLE_SIZE}) is >= data size for top classes ({df_top_n.shape[0]}). Using full top {N_TOP_CLASSES} data.")
             df_sample = df_top_n
        else:
            print(f"Sampling {SAMPLE_SIZE} rows from the top {N_TOP_CLASSES} classes for t-SNE...")
            # Stratified sampling if possible, otherwise random
            try:
                df_sample = df_top_n.groupby(target_column, group_keys=False).apply(lambda x: x.sample(min(len(x), max(1, int(SAMPLE_SIZE * len(x) / len(df_top_n))))), random_state=42)
                # Ensure we don't exceed SAMPLE_SIZE due to rounding minimums
                if len(df_sample) > SAMPLE_SIZE:
                    df_sample = df_sample.sample(n=SAMPLE_SIZE, random_state=42)
            except:
                 print("Could not perform stratified sampling, using random sampling.")
                 df_sample = df_top_n.sample(n=SAMPLE_SIZE, random_state=42)

            print(f"Sampled df shape: {df_sample.shape}")
            print(f"Number of unique classes in sample: {df_sample[target_column].nunique()}") # Should be <= N_TOP_CLASSES
    else:
        print(f"Using full dataset for the top {N_TOP_CLASSES} classes for t-SNE...")
        df_sample = df_top_n

    # --- Define Features (X) and Target (y) from the sample ---
    X = df_sample.drop(columns=[target_column])
    y = df_sample[target_column]

    # Factorize target labels for coloring - ensures consistent color mapping for the top N
    # Use the original top_n_classes list to define the categories for consistency
    y_cat = pd.Categorical(y, categories=top_n_classes, ordered=True)
    y_encoded = y_cat.codes
    uniques = top_n_classes # The unique classes are now just the top N
    n_classes = len(uniques)
    print(f"Number of unique classes in the final data for t-SNE: {n_classes}") # Should be N_TOP_CLASSES

    # --- Apply t-SNE ---
    print("\nApplying t-SNE...")
    print(f"  Perplexity: {TSNE_PERPLEXITY}")
    print(f"  Iterations: {TSNE_N_ITER}")
    print(f"  Learning Rate: {TSNE_LEARNING_RATE}")

    start_time = time.time()
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(TSNE_PERPLEXITY, X.shape[0] - 1), # Perplexity must be less than n_samples
        n_iter=TSNE_N_ITER,
        learning_rate=TSNE_LEARNING_RATE,
        n_jobs=-1 # Use all available CPU cores
    )
    X_tsne = tsne.fit_transform(X)
    end_time = time.time()
    print(f"t-SNE calculation finished in {end_time - start_time:.2f} seconds.")

    # --- Create Plot ---
    print("\nCreating plot...")
    plt.figure(figsize=(14, 10))

    # Use a suitable colormap for N_TOP_CLASSES
    # 'tab20' is good for up to 20 distinct colors
    if n_classes <= 20:
        cmap = plt.cm.get_cmap('tab20', n_classes)
    else: # Fallback if N_TOP_CLASSES > 20
        cmap = plt.cm.get_cmap('turbo', n_classes)

    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap=cmap, s=15, alpha=0.8) # Slightly larger points

    plt.title(f"t-SNE Visualization of Top {N_TOP_CLASSES} Symptoms Classes ({'Sampled ' + str(len(df_sample)) if DO_SAMPLING else 'Full Top ' + str(N_TOP_CLASSES)} Data)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.xticks([]) # Hide axis ticks for clarity
    plt.yticks([])

    # Add legend - now it should list all N_TOP_CLASSES clearly
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=uniques[i], markersize=8, markerfacecolor=cmap(i / (n_classes -1 if n_classes > 1 else 1))) for i in range(n_classes)]
    plt.legend(handles=handles, title="Diseases", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # Save the plot
    plot_filename = f"tsne_visualization_top_{N_TOP_CLASSES}_{'sampled_' + str(len(df_sample)) if DO_SAMPLING else 'full'}.png"
    plot_path = os.path.join(RESULTS_DIR, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"t-SNE plot for top {N_TOP_CLASSES} classes saved to: {plot_path}")

    # Display the plot
    plt.show()

except FileNotFoundError:
    print(f"ERROR: Data file not found at {TRAINING_DATA_PATH}")
except ImportError:
    print("ERROR: Could not import 'data_utils'. Make sure 'data_utils.py' is in the same directory or accessible in the Python path.")
except ValueError as ve:
     if "perplexity must be less than n_samples" in str(ve):
         print(f"ERROR: Perplexity ({TSNE_PERPLEXITY}) is too high for the number of samples ({X.shape[0]}). Try reducing SAMPLE_SIZE or TSNE_PERPLEXITY.")
     else:
        print(f"An error occurred: {ve}")
        import traceback
        traceback.print_exc()
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()