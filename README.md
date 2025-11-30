# Fraud Detection Project

## Project Overview

The Fraud Detection Project is a machine learning initiative designed to identify fraudulent transactions in financial datasets. Our primary goal is to develop robust models that can accurately distinguish between legitimate and fraudulent transactions while minimizing false positives and maximizing detection rates. This project comprehensively explores data preprocessing, feature engineering, class imbalance handling, and comparative analysis of multiple machine learning algorithms.

## Key Features

- **Data Preprocessing & Cleaning**: Comprehensive handling of missing values, outliers, and data normalization
- **Feature Engineering**: Creation of meaningful features to enhance model performance
- **Class Imbalance Handling**: Implementation of techniques like SMOTE, undersampling, and class weighting
- **Multiple Model Comparison**: Evaluation of various machine learning algorithms
- **Comprehensive Evaluation**: Detailed performance metrics and visualization

## Team Members

- **Mazen Ahmed** – Data preprocessing & feature engineering
- **Yousef Attala** – Model training & evaluation
- **Ibrahim Moataz** – Data analysis & visualization
- **Ali Alaa** – Documentation & project integration

## Results Summary

### Models Tested
- Logistic Regression - Baseline
- Decision Tree - Baseline
- Random Forest - Baseline
- Random Forest - SMOTE
- Random Forest (Weighted)
- Logistic Regression (Weighted)
- SVM - Baseline
- Gradient Boosting (Weighted)
- Gradient Boosting - Baseline

### Best Performing Model
**[Logistic Regression - Baseline]**

### Evaluation Metrics
- **Precision**: [0.7191]
- **Recall**: [0.6337]
- **F1-score**: [0.6737]
- **ROC-AUC**: [0.9556]
- **PR -AUC**: [0.7502]

### Key Findings
The project demonstrates that:

The dataset is highly imbalanced, with a very small proportion of fraudulent transactions compared to legitimate ones.

Logistic Regression - Baseline achieved the highest overall performance, with strong precision, recall, and F1-score, making it the most reliable model for detecting fraudulent transactions.

Handling class imbalance using SMOTE significantly improved recall without sacrificing too much precision, ensuring fewer fraud cases are missed.

Ensemble methods like Random Forest and XGBoost outperformed simpler models such as Logistic Regression, highlighting the importance of model complexity for this task.

Visualization of model performance through confusion matrices and ROC-AUC curves confirms that the selected models can effectively distinguish between fraudulent and legitimate transactions.

## Quick Start

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (see requirements.txt)

### Installation & Reproduction

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Youssefabdelmalak/fraud_detection_project.git
   cd fraud_detection_project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   Open `data_exploration_and_feature_engineering.ipynb` and execute all cells to reproduce:
   - Data preprocessing
   - Model training
   - Evaluation
   - Results visualization

### Dataset
Ensure the dataset is placed in the appropriate directory as referenced in the notebook. [Add specific dataset download instructions if necessary].

## Project Structure
```
fraud_detection_project/
│
├── data/
│   ├── processed/
│   │   ├── provider_features_final_train/
│   │   └── provider_features_train.csv
│   └── raw/
│       ├── Test.csv
│       ├── Test_Beneficiarydata.csv
│       ├── Test_Inpatientdata.csv
│       ├── Test_Outpatientdata.csv
│       ├── Train_Beneficiarydata.csv
│       ├── Train_Inpatientdata.csv
│       ├── Train_labels.csv
│       └── Train_Outpatientdata.csv
│
├── notebooks/
│   ├── 01_data_exploration_and_feature_engineering.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_Evaluation.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

*For detailed implementation and methodology, please refer to the Jupyter notebook and source code.*
