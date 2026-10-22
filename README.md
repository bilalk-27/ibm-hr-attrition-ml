# IBM HR Attrition Prediction — Machine Learning Project

[Live Demo — Try the App](https://ibm-hr-attrition-ml-2ejmwtiag3ixvyabbvvw6v.streamlit.app/)

## Project Overview

This project explores employee attrition prediction using the IBM HR Analytics Employee Attrition & Performance dataset.

The objective is to build an end-to-end Machine Learning workflow, from exploratory data analysis and preprocessing to model evaluation and deployment through an interactive Streamlit application.

**Note:** The dataset is synthetic, and this application is intended for educational and demonstration purposes only. It should not be used for real-world HR decisions.

## Technologies

- **Python:** pandas, NumPy, scikit-learn
- **Machine Learning:** Random Forest, cross-validation, hyperparameter tuning
- **Preprocessing:** ColumnTransformer, OneHotEncoder, Pipeline
- **Evaluation:** F1-score, precision, recall, confusion matrix
- **Application:** Streamlit
- **Model serialization:** joblib
- **Version control:** Git and GitHub

## Dataset

The dataset contains:

- 1,470 employee records
- 35 original columns
- A binary target variable: `Attrition` (Yes / No)
- Approximately 16% positive cases

Fourteen features were selected for the final model, including age, monthly income, overtime, job satisfaction, business travel, and years at the company.

## Methodology

1. Exploratory data analysis and data quality checks
2. Feature selection and categorical encoding
3. Stratified train / validation / test split
4. Model training and comparison
5. Five-fold stratified cross-validation
6. Classification threshold selection
7. Final evaluation on a held-out test set
8. Model serialization using joblib
9. Interactive application development with Streamlit

## Final Model Performance

The final model is a Random Forest classifier using 14 input features and a classification threshold of 0.50.

| Metric | Test Result |
|---|---:|
| F1-score (attrition class) | 0.504 |
| Precision | 0.426 |
| Recall | 0.617 |
| Accuracy | 0.806 |

The final test set contains 294 observations.

The model correctly identified 29 of the 47 attrition cases, while producing 39 false positives and 18 false negatives.

These results highlight the challenges of imbalanced classification and the trade-off between precision and recall.

## Streamlit Application

**Live application:** [HR Attrition Predictor](https://ibm-hr-attrition-ml-2ejmwtiag3ixvyabbvvw6v.streamlit.app/)

The application allows users to:

- Enter 14 characteristics of a fictional employee
- Generate a prediction using the saved Machine Learning pipeline
- Display the estimated attrition score and predicted class
- Explore how different input values affect model predictions

The estimated score should not be interpreted as a calibrated probability or a causal assessment.

## Project Structure

ibm-hr-attrition-ml/
├── main.py                 # Model training and evaluation
├── predict.py              # Prediction example
├── app.py                  # Streamlit application
├── models/
│   └── random_forest_attrition.joblib
├── requirements.txt
├── kaggle_ibm_dataset_attriction.csv
└── README.md

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python -m streamlit run app.py
```

The application should open at `http://localhost:8501`.

## Project Limitations

- The dataset is synthetic and may not reflect real employee behavior.
- Model performance is moderate, particularly in terms of precision.
- Predicted scores are not calibrated probabilities.
- Changing an input feature does not demonstrate a causal effect.
- The application is not suitable for real-world employment decisions.

## Future Improvements

- Model explainability and error analysis
- Probability calibration
- Automated testing
- Docker containerization
- Experiment tracking with MLflow
- Continuous integration and deployment
