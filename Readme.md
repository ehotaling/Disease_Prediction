# Disease Prediction from Symptoms  
**Capstone Data Mining Project – Spring 2025**

---

## Project Title  
**Disease Prediction from Symptoms using Supervised Machine Learning and LLM-enhanced Symptom Interpretation**

---

## Overview  
This project builds a multi-class disease classifier using symptom data. The trained model can predict the most
likely disease given a set of symptoms and return a recommended treatment.
As of the latest update, the CLI leverages GPT-4o mini for interpreting user-described symptoms using natural
language. This improves user interaction and expands flexibility in symptom input. This project is part of a data
mining learning initiative and includes Python scripts for data cleaning, feature selection, model training,
and prediction via a CLI interface.

---

##  Project Structure

```
Disease_Prediction/
├── data/                  # Contains raw .csv datasets
│   ├── training_data.csv
│   └── Diseases_Symptoms.csv
├── models/                # Stores pre-trained model (Random Forest)
│   └── rf_model.pkl
├── src/                   # Python scripts (all core logic lives here)
│   ├── data_utils.py
│   ├── clean_data.py
│   ├── feature_selection.py
│   ├── model_training.py
│   ├── prediction_mapping.py
│   ├── predict_cli.py
│   └── validate_mapping.py      # Validates training-treatments alignment
├── .env                   # Environment variables for API keys
├── .env.example           # Template for required environment variables
├── README.md
└── requirements.txt
```

---

##  Requirements

Install required Python packages:

```bash
pip install -r requirements.txt
```

Minimum requirements:

- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- torch
- rapidfuzz
- openai
- python-dotenv

---

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

3. **validate_mapping.py** *(optional but recommended)*

   Validates that all disease names in the training data match the canonical names in the treatment dataset.

   ```bash
   python src/validate_mapping.py
   ```

   Use this to confirm your `prognosis` labels are properly aligned with the treatment data and alias mapping logic.

4. **model_training.py**

   Trains 3 classifiers:

   - Logistic Regression
   - Random Forest  (Saved as rf_model.pkl)
   - MLP using PyTorch

   Automatically saves the trained Random Forest model in `models/rf_model.pkl`.

   ```bash
   python src/model_training.py
   ```

5. **predict_cli.py**

   Interactive command-line interface for real-time symptom prediction:

   - Accepts user input (natural language symptoms)
   - Interprets symptoms using GPT-4o mini via OpenAI API
   - Predicts disease using the pre-trained Random Forest model
   - Returns the recommended treatment using fuzzy or semantic matching

   ```bash
   python src/predict_cli.py
   ```

   **Example:**
   ```
   Enter your symptoms: I'm nauseous and have been throwing up with chills
   Interpreted symptoms: nausea, vomiting, chills
   Predicted Disease: Gastroenteritis
   Recommended Treatment: Rehydration, rest, and electrolyte replacement
   ```

---

## Dataset Summary  
- 4920 records  
- 132 binary symptom features  
- 40 unique disease classes  
- Balanced dataset with minimal symptom overlap between classes  
- Current data is clean and idealized, optimized for learning and classification, but not representative of real-world complexity

The dataset used was initially derived from publicly available Kaggle sources. Preprocessing steps included alias resolution for disease labels, column name normalization, and missing value management. A validation script was used to ensure consistent mapping between training and treatment datasets.

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
4. Integration of natural language interpretation into CLI (initially using GPT-4o, later transitioning to T5)  
5. Mapping predictions to treatment recommendations  
6. Building a real-time prediction CLI tool  
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
- Nothing Completed Yet


---

## To-Do List (Prioritized)

1. **Integrate T5 Model for Symptom Extraction**  
   Priority: Top
   - Status: Completed Evaluation. Decision made to defer T5 integration due to performance limitations compared to the GPT-4o baseline. See "Evaluation of Local T5 Model" section below for details.

2. **Refactor CLI to Support Model Choice**  
   Priority: High  
   - Allow user to select model (RF, LR, MLP) at runtime

3. **Persist All Models**  
   Priority: High  
   - Save Logistic Regression as `lr_model.pkl`  
   - Save MLP weights as `mlp_model.pth`

4. **Implement PyTorch Model Loader for CLI**  
   Priority: High  
   - Add functionality to load MLP for inference within CLI

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
- Significant Effort Required: Achieving performance comparable to the GPT-4o mini baseline would likely require substantial effort in prompt engineering, potentially exploring different local model architectures (beyond T5), or even fine-tuning a model specifically for this symptom extraction task.

### Decision: 
- Given the observed performance limitations and the significant effort required to potentially improve the local models to a reliable state, the decision was made to not proceed with the T5 integration at this time. 
- For the current scope of the project, the existing implementation using the OpenAI API (GPT-4o mini) provides superior accuracy and robustness for interpreting natural language symptom input, ensuring a more functional and reliable user experience for the CLI tool.

### Future Work Consideration: 
- While deferred for now, exploring and potentially fine-tuning local NLP models for symptom interpretation remains a valid direction for future enhancements to this project, particularly if offline capability becomes a strict requirement.

--- 

## Limitations and Future Work

### A. Current Limitations  
- Dataset is clean, perfectly labeled, and lacks symptom overlap  
- Each disease has a distinct combination of symptoms, allowing models to memorize associations without needing deep generalization  
- The models therefore perform exceptionally well (100% accuracy), but would not retain this performance on noisier, real-world datasets

### B. Why This Is Not Overfitting  
- Models are evaluated on a withheld 20% test set (never seen during training)  
- No performance degradation between training and test phases  
- There is no evidence of high training accuracy and low test accuracy  
- The feature space is linearly and non-linearly separable, enabling strong generalization *within the scope of this dataset*

### C. Future Work  
- Test pipeline against real-world or synthetically degraded datasets (with missing, noisy, or ambiguous symptom input)  
- Introduce symptom uncertainty through randomized omissions or misspellings  
- Expand to multi-label cases (multiple conditions per patient)  
- Add probabilistic outputs and confidence scoring for each prediction  
- Transition CLI to a web or mobile platform (e.g., Flask, Streamlit, or React)  
- Replace the OpenAI API dependency with a local, open-source model (T5) for offline use and full control

### D. Summary  
The models demonstrate perfect accuracy due to the dataset’s ideal structure. This does not indicate overfitting, but rather confirms that the models are effectively capturing strong class-separating patterns in the symptom features. The next phase of development will focus on generalizing this architecture to handle uncertainty, noise, and real-world variability.

### E. Note on Feature Reduction and Model Reliability
While we performed feature selection using Chi-Squared statistics and Random Forest importance to identify the most informative symptoms globally, we chose **not to reduce the input features** used in model training or prediction. This decision was made intentionally to preserve the full diversity of symptom inputs associated with each disease. 

In a multi-class classification problem like disease prediction, some features may appear **unimportant overall**, but may be **critical for correctly classifying specific diseases**, especially those with rare or unique symptoms. Reducing features based solely on global rankings risks removing these **class-specific indicators**, which could significantly degrade the accuracy for certain conditions — even if overall accuracy remains high. Additionally, feature redundancy helps absorb noise and variability in user symptom descriptions, which is particularly important in real-world deployment scenarios.

Therefore, for this phase of the project, we prioritized **maintaining full feature coverage** to ensure maximum diagnostic reliability across all classes. Future iterations may explore controlled feature reduction, but only with rigorous class-wise performance evaluation and careful testing on noisy or real-world datasets.

### F. Dimensionality and the Curse of Dimensionality
Our dataset has 132 binary symptom features, which constitutes a high-dimensional input space. However, despite the dimensionality, we do **not** suffer from the curse of dimensionality. The dataset is clean, balanced, and features are highly informative — symptoms uniquely map to diseases. Binary features, coupled with clear class boundaries and no noise, create a well-separated decision space.

The models are able to learn strong decision boundaries without the issues of sparsity or degraded distance metrics. We observe no signs of overfitting or learning instability. This means we benefit from high-dimensional input **without the typical drawbacks**, as long as the data remains clean and structured.

Should we move to a more realistic dataset in the future, dimensionality reduction and feature pruning might become more relevant tools to manage noise and improve generalization.

### G. Absence of the Long Tail Problem
Another typical challenge in multi-class classification is the **long tail problem**, where a few classes dominate the dataset while many others are severely underrepresented. This creates biased learning behavior in most models.

However, this issue does not affect our project because:
- The dataset is **balanced** — every class (disease) has approximately the same number of samples (~120)
- There are no rare diseases with too few examples to learn from
- As a result, the models are not biased toward any "head" class and perform equally well across all 40 classes

This balance makes our dataset uniquely well-suited for evaluation and benchmarking, but it also reinforces the need for future testing on **real-world datasets**, which often suffer from class imbalance and long-tail effects.

---

## Deliverables  
- Fully functioning Python CLI-based predictor with LLM symptom interpretation  
- Persisted models for Random Forest, Logistic Regression, and MLP  
- Bar charts and comparative visuals for accuracy, precision, recall, and F1  
- Presentation deck with full methodology, results, visualizations, and critical analysis  
- Clear documentation for both code and findings. 


