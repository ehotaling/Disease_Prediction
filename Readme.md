# Disease Prediction from Symptoms (Data Mining Project)

This project builds a multi-class disease classifier using symptom data. The trained model can predict the most likely disease given a set of symptoms and return a recommended treatment. As of the latest update, the CLI now leverages GPT-4o mini for interpreting user-described symptoms using natural language. This improves user interaction and expands flexibility in symptom input. This project is part of a data mining learning initiative and includes Python scripts for data cleaning, feature selection, model training, and prediction via a CLI interface.

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

##  What You'll Learn

- Binary encoding of symptoms for supervised learning
- Feature selection with Chi² and Random Forest
- Canonical mapping of disease names via normalization and alias resolution
- Multi-class classification using scikit-learn & PyTorch
- Model persistence using joblib
- CLI interface development
- Natural language interpretation using LLMs (GPT-4o mini)
- Validation of dataset consistency across multiple sources

---

##  Future Ideas

- Turn `predict_cli.py` into a web app using Streamlit or Flask
- Add support for probabilistic confidence scores
- Extend to include real patient data and lab results
- Export predictions and logs to a file for audit/history
- Incorporate symptom timelines or severity scores

