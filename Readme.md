# Disease Prediction from Symptoms  
**Capstone Data Mining Project – Spring 2025**

---

## Project Title  
**Disease Prediction from Symptoms using Supervised Machine Learning and LLM-enhanced Symptom Interpretation**

---

## Overview  
This project builds a multi-class disease classifier using symptom data. The trained model can predict the most
likely disease given a set of symptoms and return a recommended treatment, dynamically generated via the OpenAI API based on the predicted disease.
As of the latest update, the CLI leverages GPT-4o mini for interpreting user-described symptoms using natural
language. This improves user interaction and expands flexibility in symptom input. This project is part of a data
mining learning initiative and includes Python scripts for data cleaning, feature selection, model training,
and prediction via a CLI interface.

---

## Dataset Link
Before running the project, you must manually download the dataset used for training and prediction. 
This dataset is not included in the repository due to its size and licensing.


##  Project Structure

```
Disease_Prediction/
├── data/                    # Contains raw .csv datasets
│   └── training_data.csv    # Contains training data
├── models/                  # Stores pre-trained models
│   ├── rf_model.pkl
│   ├── lr_model.pkl
│   └── mlp_model.pth
├── src/                     # Python scripts (all core logic lives here)
│   ├── data_utils.py
│   ├── clean_data.py
│   ├── feature_selection.py
│   ├── model_training.py
│   └── predict_cli.py       # Core prediction and API interaction logic
├── .env                     # Environment variables for API keys
├── .env.example             # Template for required environment variables
├── README.md
├── requirements-cpu.txt     # For CPU-only environments
├── requirements-gpu.txt     # For CUDA-enabled GPU systems
└── requirements-mac-arm.txt # Optimized for macOS ARM (M1/M2/M3)
```

---

## Requirements

Install required Python packages depending on your system architecture and available hardware:

1.  ** CPU Users (no compatible GPU):**
    Use this if you’re on a typical desktop or server without a CUDA-capable NVIDIA GPU.
    ```bash
    pip install -r requirements-cpu.txt
    ```
   - Uses torch==2.2.2+cpu and related CPU-only builds via PyTorch’s index. 

2.  **GPU Users (CUDA 12.8+, NVIDIA GPU):**
    Use this if you’re training or running models on a CUDA-capable GPU with up-to-date drivers.
    ```bash
    pip install -r requirements-gpu.txt
    ```
    - Uses nightly builds like torch==2.8.0.dev...+cu128.
    - Pulls packages from https://download.pytorch.org/whl/nightly/cu128.
    - Ensure your GPU and driver support CUDA 12.8 and that your system is configured properly for PyTorch GPU usage.

3. **macOS ARM (Apple M1/M2/M3):**
    For Macs with Apple Silicon. These packages are built natively for ARM64.
    
    ```bash
    pip install -r requirements-mac-arm.txt
    ```
   - Uses standard PyPI index (no extra index required).
   - Installs native torch==2.2.2 and friends compatible with macOS ARM wheels.


**Minimum requirements (common to both):**

* pandas
* numpy
* scikit-learn
* matplotlib
* joblib
* torch (CPU or GPU version)
* rapidfuzz
* openai
* python-dotenv
* tabulate *(Added dependency for markdown table printing)*


---

###  NumPy Compatibility Notice

All `requirements-*.txt` files pin NumPy to `<2.0.0` for stability and compatibility across platforms.

**Why this is necessary:**

- NumPy 2.0 introduces a breaking change to its internal C-extension API (`_ARRAY_API`).
- Many libraries in this project (including **PyTorch**, **scikit-learn**, and **joblib**) are still built against NumPy 1.x and may crash or fail to import if NumPy 2.x is installed.
- This affects **all platforms**, including Windows, macOS (ARM and Intel), and Linux.

**Example error (macOS ARM):**

A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2...


**Temporary Fix:**
Until full ecosystem support for NumPy 2.0 is released, we have pinned:
```txt
numpy<2.0.0 
```
in requirements-gpu.txt, requirements-cpu.txt, and requirements-mac-arm.txt.
This ensures consistent behavior across devices and avoids cryptic import-time crashes.



##  Environment Variables

Create a `.env` file at the root of the project with the following contents:

```
# .env.example
OPENAI_API_KEY=your_openai_api_key_here
```

---

##  Order of Execution

Here is the recommended order for running the scripts:

1. **clean_data.py** *(optional for EDA)*
   
   Run this to explore the dataset and confirm that your files are formatted and cleaned properly.

   ```bash
   python src/clean_data.py
   ```

2. **feature_selection.py** *(optional)*

   ```bash
   python src/feature_selection.py
   ```

   Identifies the most predictive symptoms using:

   - Chi-Squared Scores
   - Random Forest Feature Importances


3. **model_training.py**

   Trains 3 classifiers:

   - Logistic Regression
   - Random Forest
   - MLP using PyTorch

   Automatically saves the trained models (`rf_model.pkl`, `lr_model.pkl`, `mlp_model.pth`) in the `models/` directory.

   ```bash
   python src/model_training.py
   ```

4. **predict_cli.py**

   Interactive command-line interface for real-time symptom prediction:

   - Accepts user input (natural language symptoms).
   - Interprets symptoms using GPT-4o mini via OpenAI API.
   - Allows user to select prediction model (RF, LR, MLP).
   - Predicts disease using the selected pre-trained model.
   - **Generates a concise treatment recommendation for the predicted disease using the OpenAI API.**
   - **Important Disclaimer:** Treatment recommendations are AI-generated and *not* a substitute for professional medical advice.

   ```bash
   python src/predict_cli.py
   ```

   **Example:**
   ```
   Enter model you want to use (RF/LR/MLP): RF
   Random Forest model loaded successfully.
   Enter your symptoms: I'm nauseous and have been throwing up with chills
   Interpreted symptoms: nausea, vomiting, chills

   Prediction Results:
   -------------------
   Predicted Disease: Gastroenteritis
   Recommended Treatment: [AI-generated treatment suggestion, e.g., Focus on rehydration with water or electrolyte solutions. Rest is important. Over-the-counter medications may help with nausea, but consult a doctor if symptoms persist or worsen.]
   ```

---

## Dataset Summary

### Download Required

Before using this project, you must manually download the dataset from Kaggle:

**Dataset URL:**  
[https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset?resource=download](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset?resource=download)

After downloading:

1. Unzip the file.
2. Rename the file to `training_data.csv` if needed.
3. Create a `data/` directory in the project root if it doesn't already exist.
4. Move the file to the project’s `data/` directory:
    `Disease_Prediction/data/training_data.csv`

> This file is required for all stages of the pipeline, including data cleaning, training, and prediction. It is not included in the repository due to licensing and file size.


The current dataset, sourced from Kaggle (https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset?resource=download), is an artificially generated collection designed to reflect potential real-world disease occurrence probabilities and symptom co-occurrences.

Key characteristics:
- **Initial Size:** ~247,000 records across 773 unique disease classes.
- **Features:** 377 binary symptom features (1 indicating presence, 0 absence).
- **Filtering:** Classes with fewer than 3 samples were removed prior to modeling to enable stratified train/test splitting, resulting in **748 unique disease classes** used for training and evaluation.
- **Nature:** Contains varying combinations of symptoms for diseases and exhibits significant **class imbalance** (long tail effect), unlike the previous idealized dataset. This presents a more realistic modeling challenge.
- **Baseline Performance:** Initial model training (LR, RF, MLP) yields baseline accuracies in the mid-80s, reflecting the increased complexity compared to the previous dataset where 100% accuracy was achieved due to its separable nature.

This dataset replaces the previous smaller, balanced dataset. Consequently, prior preprocessing steps like alias resolution and validation against a separate treatment mapping file are no longer applicable.

---

## Motivation  
- Accelerate symptom-based diagnosis using intelligent automation  
- Enable early intervention and reduce unnecessary testing  
- Provide a foundation for scalable medical triage tools  
- Gain practical experience in supervised learning, NLP integration, and end-to-end model deployment

---

## Key Objectives
1. Data preprocessing and normalization 
2. Feature selection using Chi-Squared scores and model-based importances 
3. Model training and evaluation (Random Forest, Logistic Regression, MLP) 
4. Integration of natural language interpretation into CLI (using GPT-4o; T5 evaluation completed & deferred)
5. Generating treatment recommendations dynamically via OpenAI API based on predictions 
6. Building a real-time prediction CLI tool *(Updated features)*
7. Presenting the project in a professional, academic format with visuals and metrics

---

## Completed Milestones Midpoint Presentation 
- Data loaded, cleaned, and preprocessed  
- EDA completed, including symptom frequency analysis and class distribution  
- Chi-Squared and Random Forest feature selection conducted  
- Models trained and evaluated: Random Forest, Logistic Regression, and PyTorch-based MLP  
- All models achieved 100% test accuracy due to dataset separability  
- CLI built with GPT-4o mini for natural language symptom interpretation  
- Predictions mapped to recommended treatments  
- Model training and evaluation results plotted (accuracy, precision, recall, F1)  
- Model persistence implemented for Random Forest (`rf_model.pkl`)  
- Midpoint presentation prepared with live demo

---

## Completed Milestones Final Presentation
Work completed since the midpoint presentation includes:

- **Environment & Dependencies:** Resolved PyTorch/CUDA compatibility issues, enabling GPU acceleration for MLP training (`Using device: cuda`). Updated requirements into separate `requirements-gpu.txt` and `requirements-cpu.txt`.
- **Code Stabilization:** Addressed deprecation warnings (sklearn, pandas) and PyTorch warnings in `model_training.py`.
- **NLP Evaluation:** Evaluated local T5 model for symptom interpretation; decided to retain OpenAI API (GPT-4o mini) for better performance and reliability (See "Evaluation of Local T5 Model" section).
- **Model Persistence & Selection:**
    - Refactored `model_training.py` to save all three models (Logistic Regression, Random Forest, PyTorch MLP).
    - Refactored `predict_cli.py` to allow user selection of the desired model (RF/LR/MLP) at runtime.
    - Implemented PyTorch model loading (`state_dict`) and inference logic within `predict_cli.py`.
- **New Dataset Integration:**
    - Integrated a new, larger dataset (~247k rows, 773 initial diseases, 377 symptoms).
    - Refactored `data_utils.py` for loading and basic cleaning of the new dataset format.
    - Implemented filtering in `model_training.py` to remove classes with < 3 samples, resulting in 748 classes used for modeling and enabling stratified splitting.
- **Baseline Model Training (New Dataset):**
    - Successfully executed the refactored `model_training.py` pipeline on the full (filtered) new dataset.
    - Trained LR, RF, and MLP models, achieving baseline accuracies in the mid-80s.
    - Generated and saved model comparison metrics (CSV table, markdown table) and visualization plots (matplotlib table, bar charts) to a new `results/` directory.
- **Planned Architecture Change:** Decided to switch from static treatment lookup to dynamic generation via OpenAI API (to be implemented in `predict_cli.py`). Relevant Readme sections updated to reflect this plan.

---

## To-Do List (Prioritized)

1. **Integrate T5 Model for Symptom Extraction**  
   Priority: Top
   - Status: Completed Evaluation. Decision made to defer T5 integration due to performance limitations compared to the GPT-4o baseline. See "Evaluation of Local T5 Model" section below for details.

2. **Refactor CLI to Support Model Choice**  
   Priority: High
   - Status: **Completed.** CLI now prompts user to select RF, LR, or MLP model at runtime.

3. **Persist All Models**  
   Priority: High  
   - Status: **Completed.** `model_training.py` now saves LR (`lr_model.pkl`) and MLP (`mlp_model.pth`) alongside RF model.

4. **Implement PyTorch Model Loader for CLI**  
   Priority: High  
   - Status: **Completed.** `predict_cli.py` now correctly loads the saved MLP model (`mlp_model.pth`) and uses it for prediction when selected.

5. **Expand Feature Selection**  
   Priority: Medium  
   - Add Mutual Information and Recursive Feature Elimination (RFE)  
   - Create merged importance rankings across methods

6. **Improve Model Training**  
   Priority: Medium  
   - Apply k-fold cross-validation  
   - Tune hyperparameters for each model (e.g., grid search or random search)

7. **Enhance Evaluation**  
   Priority: Low  
   - Add confusion matrices, ROC curves, top-k (e.g., top-3) predictions  
   - Conduct per-class analysis to assess where models perform differently

8. **Finalize Presentation Visuals**  
   Priority: High  
   - Add system architecture diagrams  
   - Add saved feature plots and model comparison visuals  
   - Add CLI flowchart

9. **Polish Final Presentation**  
   Priority: High  
   - Improve "Since Last Time" slide  
   - Streamline walkthrough and structure of final deck  
   - Ensure CLI demo is clear and illustrative

10. **Create Limitations and Future Work Section**  
    Priority: Top  
    - Explicitly address dataset limitations and discuss realistic future extensions

---

## Evaluation of Local T5 Model for Symptom Interpretation

### Goal: 
- As part of exploring alternatives to external API calls, we investigated the integration of a local HuggingFace T5 model (specifically testing t5-small and t5-base) to replace the OpenAI GPT-4o mini for natural language symptom interpretation within the predict_cli.py tool. 
- The goal was to enable offline functionality and remove the dependency on the OpenAI API for this component.

### Approach: 
- The T5 model was integrated into the CLI. User input was processed using various prompts designed to instruct the T5 model (e.g., "extract medical symptom keywords...") to identify relevant terms from the user's description. 
- The model's output was then parsed, and the potential symptoms were matched against the project's canonical symptom list (feature_cols) using a combination of direct string matching and rapidfuzz fuzzy matching (WRatio score > 80) to handle minor variations.

### Findings & Challenges:
- Inconsistent Performance: Both t5-small and the larger t5-base models exhibited inconsistent performance in reliably extracting all relevant symptoms from user descriptions. Tests showed instances where obvious keywords like 'cough' or 'fever' were missed, even after correcting initial implementation bugs.
- Sensitivity to Phrasing: The models proved highly sensitive to minor variations in user input phrasing (e.g., "fever" vs. "a fever"). This lack of robustness would lead to an unreliable user experience.
- Extraction Failures: Even with t5-base and refined matching logic, the model failed to interpret relatively straightforward symptom combinations (e.g., unable to extract keywords from "dizzy, swollen legs, hungry" in final tests).
- Significant Effort Required: Achieving performance comparable to the GPT-4o mini baseline would likely require significant effort in prompt engineering, potentially exploring different local model architectures (beyond T5), or even fine-tuning a model specifically for this symptom extraction task.

### Decision: 
- Given the observed performance limitations and the significant effort required to potentially improve the local models to a reliable state, the decision was made to not proceed with the T5 integration at this time. 
- For the current scope of the project, the existing implementation using the OpenAI API (GPT-4o mini) provides superior accuracy and robustness for interpreting natural language symptom input, ensuring a more functional and reliable user experience for the CLI tool.

### Future Work Consideration: 
- While deferred for now, exploring and potentially fine-tuning local NLP models for symptom interpretation remains a valid direction for future enhancements to this project, particularly if offline capability becomes a strict requirement.

---

## Limitations and Future Work

### A. Current Limitations
- **Artificial Dataset:** While large (~247k records, 773 initial diseases, 377 symptoms), the dataset used is artificially generated. It may lack the noise, nuances, missing values, and complex symptom correlations found in real-world clinical data. Potential biases from the generation process are unknown.
- **Class Imbalance:** The dataset exhibits a significant class imbalance (long tail problem), even after filtering out the rarest classes (< 3 samples). This can affect model performance, particularly for less frequent diseases.
- **Baseline Performance:** Current models (LR, RF, MLP) achieve baseline accuracies in the mid-80s. While reasonable for this complex task without tuning, there is significant room for improvement.
- **Symptom Interpretation:** Relies on OpenAI API (GPT-4o mini), requiring an internet connection and API key. The quality of interpretation can impact prediction accuracy.
- **Treatment Generation:** The planned dynamic treatment generation via OpenAI API needs careful implementation and validation due to the sensitive nature of medical advice, and will carry strong disclaimers.

### B. Monitoring Overfitting
- Unlike the previous idealized dataset which yielded 100% test accuracy with no signs of overfitting, the current models train on a complex dataset where overfitting is a potential concern.
- Current evaluation is based on a single train-test split. While the test performance (~83-87%) doesn't immediately suggest *severe* overfitting occurred within the tested 20 epochs for the MLP, rigorous monitoring was not implemented.
- Future work (Task #6/#7) should incorporate validation sets and monitoring during training (e.g., plotting training vs. validation loss/accuracy) to detect overfitting and potentially implement techniques like early stopping or regularization.

### C. Future Work
- **Test on Real-World Data:** Evaluate the pipeline using real clinical datasets (if available) to assess true performance.
- **Address Class Imbalance:** Implement techniques specifically designed for imbalanced datasets (e.g., resampling methods like SMOTE, class-weighted loss functions).
- **Improve Model Training (Task #6):** Apply k-fold cross-validation for more robust evaluation. Tune hyperparameters for each model using techniques like grid search or randomized search. Explore more complex model architectures.
- **Enhance Evaluation (Task #7):** Generate confusion matrices, ROC curves (potentially using one-vs-rest), and calculate top-k accuracy. Conduct detailed per-class analysis to understand performance on specific diseases, especially rare ones.
- **Expand Feature Selection (Task #5):** Implement and compare additional feature selection methods (Mutual Information, RFE) and analyze their impact on performance and training time.
- **Probabilistic Outputs:** Modify models/output to provide prediction probabilities or confidence scores.
- **Multi-Label Classification:** Extend the system to handle cases where a patient might present with symptoms indicative of multiple simultaneous conditions.
- **UI Development:** Transition the CLI to a more user-friendly web or mobile platform (e.g., using Flask, Streamlit, or React).
- **Local NLP Exploration (Deferred):** Revisit the use of local open-source models for symptom interpretation if offline capability becomes critical and more powerful local models or fine-tuning techniques are explored (see "Evaluation of Local T5 Model" section).

### D. Summary 
- The models demonstrate reasonable baseline performance (accuracies ~83-87%, macro F1 ~80-84%) on a large, complex, and imbalanced dataset featuring 748 disease classes and 377 symptoms.
- This indicates the models are learning meaningful patterns beyond simple memorization seen in the previous idealized dataset.
- Current development focuses on integrating dynamic treatment generation and establishing robust training/evaluation pipelines, while future work will target performance improvement through feature engineering, model tuning, and advanced evaluation techniques.

### E. Note on Feature Reduction and Model Reliability 
- With 377 symptom features, feature selection/reduction (Task #5) becomes a more relevant consideration than with the previous smaller dataset, potentially offering benefits in training time and model simplicity.
- However, the core principle remains: care must be taken to ensure that reducing features based on global importance metrics does not inadvertently remove symptoms critical for identifying specific, potentially rare, diseases.
- Any feature reduction strategy should be evaluated rigorously, including its impact on per-class performance metrics, before being fully adopted. For the initial baseline, full feature coverage was maintained.

### F. Dimensionality
- The current dataset utilizes **377 binary symptom features**, creating a higher-dimensional input space compared to the previous dataset (132 features).
- While higher dimensionality can sometimes lead to the "curse of dimensionality" (sparsity, degraded distance metrics), the baseline models (LR, RF, MLP) achieved reasonable performance (~83-87% accuracy).
- This suggests that while the dimensionality is significant, it is not currently preventing the models from learning effectively on this dataset. Techniques like feature reduction or dimensionality reduction (e.g., PCA, though less common for binary features) might become more relevant if performance plateaus or during hyperparameter tuning.

### G. Presence of the Long Tail Problem 
- Unlike the previous balanced dataset, the current dataset **exhibits a significant long tail problem**, meaning some disease classes have many samples while a large number of classes have very few.
- This was confirmed during data preparation, where **25 classes were removed entirely** because they had fewer than 3 samples, making stratified splitting impossible.
- The remaining **748 classes are still likely imbalanced** to varying degrees, reflecting the dataset description regarding real-world occurrence probability.
- **Implications:** Class imbalance can bias models towards predicting more frequent classes and can lead to poor performance (especially low recall) on rare classes, even if overall accuracy appears high. Macro-averaged metrics (like those used in the comparison table) are sensitive to this imbalance.
- **Mitigation:** The filtering step addresses the most extreme cases. Future work should explicitly consider techniques to handle class imbalance during training or evaluation (see Future Work section).

---

## Deliverables  
- Fully functioning Python CLI-based predictor with LLM symptom interpretation and dynamic, AI-generated treatment recommendations (via OpenAI API). 
- Persisted models for Random Forest, Logistic Regression, and MLP  
- Bar charts and comparative visuals for accuracy, precision, recall, and F1  
- Presentation deck with full methodology, results, visualizations, and critical analysis  
- Clear documentation for both code and findings. 


