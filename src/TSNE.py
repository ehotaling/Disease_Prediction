import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import data_utils  # Import the utility functions from your data_utils.py file
import time  # To measure execution time

# --- Configuration ---
TRAINING_DATA_PATH = "../data/training_data.csv"
RESULTS_DIR = "../results"  # Directory to save the plot and textual data
DO_SAMPLING = False  # Set to True to run on a sample, False for full data (False is SLOW!)
SAMPLE_SIZE = 5000  # Number of samples if DO_SAMPLING is True
MIN_SAMPLES_PER_CLASS = 3  # Minimum samples per class to keep (consistent with model_training.py)
N_TOP_CLASSES = 10  # Number of top classes to visualize. Adjust this as needed.
TSNE_PERPLEXITY = 30  # Typical value, adjust as needed
TSNE_N_ITER = 1000  # Number of iterations for optimization
TSNE_LEARNING_RATE = 200  # Typical value, adjust as needed # or 'auto' for newer sklearn
LOKY_MAX_CPU_COUNT = "8"  # Set as a string for environment variable
os.environ['LOKY_MAX_CPU_COUNT'] = LOKY_MAX_CPU_COUNT

# --- Ensure results directory exists ---
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created directory: {RESULTS_DIR}")

# --- Load and Prepare Data ---
print("Loading and cleaning dataset...")
try:
    # Use the function from data_utils.py
    df = data_utils.load_and_clean_data(TRAINING_DATA_PATH, 'training')  #
    print(f"Dataset loaded. Initial shape: {df.shape}")

    target_column = 'prognosis'  # Defined in data_utils cleaning

    # --- Filtering rare classes (consistent with model_training.py) ---
    print(f"Filtering classes with < {MIN_SAMPLES_PER_CLASS} samples...")  #
    class_counts = df[target_column].value_counts()
    classes_to_keep = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
    df_filtered = df[df[target_column].isin(classes_to_keep)].copy()
    print(f"Shape after filtering rare classes: {df_filtered.shape}")
    print(f"Number of unique classes after filtering: {df_filtered[target_column].nunique()}")

    # --- Identify and Filter for Top N Classes ---
    print(f"\nIdentifying top {N_TOP_CLASSES} most frequent classes...")
    top_n_classes = df_filtered[target_column].value_counts().nlargest(N_TOP_CLASSES).index.tolist()
    print(f"Top {N_TOP_CLASSES} classes: {top_n_classes}")

    # Filter the DataFrame to keep only rows belonging to the top N classes
    df_top_n = df_filtered[df_filtered[target_column].isin(top_n_classes)].copy()
    print(f"Shape after filtering for top {N_TOP_CLASSES} classes: {df_top_n.shape}")

    # --- Optional Sampling (Applied to the Top N subset) ---
    if DO_SAMPLING:
        if SAMPLE_SIZE >= df_top_n.shape[0]:
            print(
                f"Sample size ({SAMPLE_SIZE}) is >= data size for top classes ({df_top_n.shape[0]}). Using full top {N_TOP_CLASSES} data.")
            df_sample = df_top_n
        else:
            print(f"Sampling {SAMPLE_SIZE} rows from the top {N_TOP_CLASSES} classes for t-SNE...")
            # Stratified sampling if possible, otherwise random
            try:
                # Ensure enough samples per group for stratification logic
                min_group_size_for_strat = 1  # Or some other small number
                df_top_n_counts = df_top_n[target_column].value_counts()
                eligible_groups = df_top_n_counts[df_top_n_counts >= min_group_size_for_strat].index

                if len(eligible_groups) < df_top_n[target_column].nunique():
                    print("Some classes have too few samples for stratified sampling, using random sampling for all.")
                    raise ValueError("Stratification not possible for all groups.")

                df_sample = df_top_n.groupby(target_column, group_keys=False).apply(
                    lambda x: x.sample(
                        min(len(x), max(1, int(SAMPLE_SIZE * len(x) / len(df_top_n)))) if len(x) > 0 else x),
                    random_state=42
                )
                # Ensure we don't exceed SAMPLE_SIZE due to rounding minimums
                if len(df_sample) > SAMPLE_SIZE:
                    df_sample = df_sample.sample(n=SAMPLE_SIZE, random_state=42)
                if len(df_sample) == 0 and SAMPLE_SIZE > 0 and len(
                        df_top_n) > 0:  # Handle case where sampling results in empty df
                    print(
                        "Warning: Stratified sampling resulted in an empty DataFrame. Falling back to random sampling.")
                    df_sample = df_top_n.sample(n=SAMPLE_SIZE, random_state=42)

            except Exception as e:  # Catch broader exceptions during sampling too
                print(f"Could not perform stratified sampling due to: {e}. Using random sampling.")
                df_sample = df_top_n.sample(n=SAMPLE_SIZE, random_state=42)

            print(f"Sampled df shape: {df_sample.shape}")
            print(f"Number of unique classes in sample: {df_sample[target_column].nunique()}")
    else:
        print(f"Using full dataset for the top {N_TOP_CLASSES} classes for t-SNE ({df_top_n.shape[0]} samples)...")
        df_sample = df_top_n

    if df_sample.empty:
        print("ERROR: DataFrame for t-SNE is empty. Exiting.")
        exit()

    # --- Define Features (X) and Target (y) from the sample ---
    X = df_sample.drop(columns=[target_column])
    y = df_sample[target_column]  # This is a Series with original class names

    # Factorize target labels for coloring - ensures consistent color mapping for the top N
    # Use the original top_n_classes list to define the categories for consistency
    y_cat = pd.Categorical(y, categories=top_n_classes, ordered=True)
    y_encoded = y_cat.codes  # These are the integer encoded labels
    uniques = top_n_classes  # The unique classes are now just the top N
    n_classes = len(uniques)
    print(f"Number of unique classes in the final data for t-SNE: {n_classes}")

    # --- Apply t-SNE ---
    print("\nApplying t-SNE...")
    print(f"  Perplexity: {TSNE_PERPLEXITY}")
    print(f"  Iterations: {TSNE_N_ITER}")  # Note: scikit-learn uses max_iter
    print(f"  Learning Rate: {TSNE_LEARNING_RATE}")

    start_time = time.time()
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(TSNE_PERPLEXITY, X.shape[0] - 1),  # Perplexity must be less than n_samples
        max_iter=TSNE_N_ITER,  # Updated from n_iter
        learning_rate=TSNE_LEARNING_RATE,  # Consider 'auto' for sklearn >= 1.2
        n_jobs=-1  # Use all available CPU cores
    )
    X_tsne = tsne.fit_transform(X)  # This returns a NumPy array
    end_time = time.time()
    print(f"t-SNE calculation finished in {end_time - start_time:.2f} seconds.")

    # --- Textual Representation of t-SNE results ---
    print("\n--- Textual t-SNE Results ---")
    tsne_results_df = pd.DataFrame(data=X_tsne, columns=['tsne_component_1', 'tsne_component_2'])
    # Ensure 'y' (original class names) has the same index as X_tsne if X was from df_sample
    tsne_results_df['prognosis'] = y.values  # y is already aligned with df_sample
    tsne_results_df['prognosis_encoded'] = y_encoded

    # Calculate centroids
    class_centroids = tsne_results_df.groupby('prognosis')[
        ['tsne_component_1', 'tsne_component_2']].mean().reset_index()
    # Order centroids by the top_n_classes list for consistency
    class_centroids['prognosis'] = pd.Categorical(class_centroids['prognosis'], categories=top_n_classes, ordered=True)
    class_centroids = class_centroids.sort_values('prognosis').reset_index(drop=True)

    print("\nClass Centroids in t-SNE space:")
    print(class_centroids.to_string())  # Print full df without truncation

    # Save the full t-SNE coordinates and labels to a CSV
    tsne_coords_filename = f"tsne_coordinates_top_{N_TOP_CLASSES}_classes.csv"
    tsne_coords_path = os.path.join(RESULTS_DIR, tsne_coords_filename)
    try:
        tsne_results_df.to_csv(tsne_coords_path, index=False)
        print(f"\nFull t-SNE coordinates saved to: {tsne_coords_path}")
    except Exception as e_csv:
        print(f"Error saving t-SNE coordinates to CSV: {e_csv}")

    # --- Create Plot ---
    print("\nCreating plot...")
    plt.figure(figsize=(14, 10))

    # Determine Colormap name
    if n_classes <= 20:
        cmap_name = 'tab20'
    else:  # Fallback if N_TOP_CLASSES > 20
        cmap_name = 'turbo'

    active_cmap = plt.colormaps.get_cmap(cmap_name)

    scatter = plt.scatter(
        tsne_results_df['tsne_component_1'],  # Use data from DataFrame for consistency
        tsne_results_df['tsne_component_2'],
        c=tsne_results_df['prognosis_encoded'],  # Use encoded labels from DataFrame
        cmap=cmap_name,
        s=15,
        alpha=0.8
    )

    plt.title(
        f"t-SNE Visualization of Top {N_TOP_CLASSES} Symptoms Classes ({'Sampled ' + str(len(df_sample)) if DO_SAMPLING else 'Full Top ' + str(N_TOP_CLASSES)} Data)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.xticks([])
    plt.yticks([])

    if n_classes > 0:
        norm_factor = (n_classes - 1) if n_classes > 1 else 1
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=uniques[i],  # uniques is top_n_classes
                              markersize=8,
                              markerfacecolor=active_cmap(i / norm_factor if norm_factor > 0 else 0))
                   for i in range(n_classes)]  # Iterate up to n_classes
        plt.legend(handles=handles, title="Diseases", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        print("No classes to plot in legend.")

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_filename = f"tsne_visualization_top_{N_TOP_CLASSES}_classes.png"
    plot_path = os.path.join(RESULTS_DIR, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    print(f"t-SNE plot for top {N_TOP_CLASSES} classes saved to: {plot_path}")

    plt.show()

except FileNotFoundError:
    print(f"ERROR: Data file not found at {TRAINING_DATA_PATH}")
except ImportError as ie:
    print(
        f"ERROR: Could not import 'data_utils'. Make sure 'data_utils.py' is in the correct path and dependencies are installed: {ie}")
except ValueError as ve:
    if "perplexity must be less than n_samples" in str(ve):
        print(
            f"ERROR: Perplexity ({TSNE_PERPLEXITY}) is too high for the number of samples ({X.shape[0] if 'X' in locals() else 'N/A'}). Try reducing SAMPLE_SIZE or TSNE_PERPLEXITY, or check data filtering steps.")
    else:
        print(f"A ValueError occurred: {ve}")
        import traceback

        traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback

    traceback.print_exc()