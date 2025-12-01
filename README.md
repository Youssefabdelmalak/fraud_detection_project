# Healthcare Provider Fraud Detection Project

## Project Overview

The Healthcare Provider Fraud Detection Project is a machine learning initiative designed to identify fraudulent healthcare providers in Medicare claims data. This work was developed for the Centers for Medicare & Medicaid Services (CMS) to assist investigators in surfacing high-risk providers for further review. Our primary goal is to develop robust models that can accurately distinguish between legitimate and fraudulent providers while minimizing false positives and maximizing detection rates. The project covers data preprocessing, feature engineering, class imbalance handling, and comparative analysis of multiple machine learning algorithms.

**Problem Statement:** Healthcare fraud costs the U.S. healthcare system over $68 billion annually. CMS can currently only investigate a small fraction of suspicious cases, allowing many fraudulent activities to go undetected. Existing systems rely on basic rule-based methods that capture obvious patterns but fail to identify more sophisticated fraud schemes.

**Objective:** Build an end-to-end fraud detection pipeline capable of:
- Detecting fraudulent providers from multi-table claims data
- Handling severe class imbalance (approximately 9.35% of providers are labeled fraudulent)
- Providing explainable predictions for investigators and regulators
- Demonstrating business value by prioritizing high-risk providers effectively

**Dataset:** The project uses the Healthcare Provider Fraud Detection dataset from Kaggle, containing anonymized Medicare data with 5,410 providers, 138,556 beneficiaries, and over 558,000 claims (inpatient and outpatient).

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

**Logistic Regression - Baseline**

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision** | 0.7191 | 71.91% of flagged providers are fraudulent |
| **Recall** | 0.6337 | 63.37% of fraudulent providers detected |
| **F1-Score** | 0.6737 | Balanced performance metric |
| **ROC-AUC** | 0.9556 | Excellent discriminatory ability |
| **PR-AUC** | 0.7502 | Strong performance on imbalanced data |

### Models Evaluated

11 model configurations were tested across 5 algorithm types:
- Logistic Regression (Baseline, Weighted, Balanced)
- Random Forest (Baseline, Robust, SMOTE, Weighted)
- Decision Tree (Baseline)
- Gradient Boosting (Baseline, Weighted)
- SVM (Baseline)

### Key Findings

1. **Class Imbalance:** The dataset is highly imbalanced with 9.35% of providers labeled as fraudulent (506 out of 5,410 providers).

2. **Best Model:** Logistic Regression - Baseline achieved the highest F1-score (0.6737) with a good balance of precision (0.7191) and recall (0.6337), making it the most reliable model for detecting fraudulent healthcare providers.

3. **Fraud Patterns:** Fraudulent providers exhibit distinct patterns:
   - Approximately **5.4x higher total inpatient claim amounts** compared to legitimate providers
   - Higher patient counts
   - More diverse geographic coverage

4. **Imbalance Handling:** SMOTE and class weighting significantly improved recall (up to 0.91) but at the cost of precision, demonstrating the precision-recall trade-off in fraud detection.

5. **Business Impact:** The model successfully identifies 63.37% of fraudulent providers while maintaining 71.91% precision, meaning most flagged providers are indeed fraudulent and warrant investigation.

## Reproduction Instructions

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook
- Required Python packages (see `requirements.txt`)

### Step-by-Step Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Youssefabdelmalak/fraud_detection_project.git
   cd fraud_detection_project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Required packages include: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, and jupyter.

3. **Download Dataset**
   
   Download the dataset from Kaggle: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
   
   **Required Files:**
   - `Train_Beneficiarydata.csv` - Patient demographics and chronic conditions
   - `Train_Inpatientdata.csv` - Inpatient hospital claims
   - `Train_Outpatientdata.csv` - Outpatient claims
   - `Train_labels.csv` - Provider-level fraud labels
   
   Place all files in the `data/raw/` directory.

4. **Run the Notebooks**
   
   Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   
   Execute notebooks in the following order:
   
   **Notebook 1:** `01_data_exploration_and_feature_engineering.ipynb`
   - Loads and explores the data
   - Performs data quality checks
   - Conducts exploratory data analysis
   - Creates provider-level features (50 features)
   - Selects final features for modeling (43 features)
   - Saves processed data to `data/processed/provider_features_final_train.csv`
   
   **Notebook 2:** `02_modeling.ipynb`
   - Loads processed features
   - Splits data into training and test sets
   - Trains 11 different model configurations
   - Compares model performance
   - Selects best model (Logistic Regression - Baseline)
   
   **Notebook 3:** `03_Evaluation.ipynb`
   - Evaluates best model on test set
   - Generates confusion matrices
   - Creates ROC and Precision-Recall curves
   - Performs cost-based analysis
   - Conducts error analysis (false positives and false negatives)

### Expected Outputs

After running all notebooks, you should have:
- Processed feature files in `data/processed/`
- Model evaluation results with metrics
- Visualizations (confusion matrices, ROC curves, PR curves)
- Error analysis identifying false positives and false negatives

### Project Structure

```
fraud_detection_project/
│
├── data/
│   ├── processed/
│   │   ├── provider_features_final_train.csv
│   │   └── provider_features_train.csv
│   └── raw/
│       ├── Train_Beneficiarydata.csv
│       ├── Train_Inpatientdata.csv
│       ├── Train_Outpatientdata.csv
│       └── Train_labels.csv
│
├── notebooks/
│   ├── 01_data_exploration_and_feature_engineering.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_Evaluation.ipynb
│
├── reports/
│   └── technical_report.md
│
├── README.md
└── requirements.txt
```

### Troubleshooting

- **Missing data files:** Ensure all CSV files are in `data/raw/` directory
- **Import errors:** Verify all packages are installed: `pip install -r requirements.txt`
- **Path errors:** Run notebooks from the project root directory or adjust paths accordingly

---

**For detailed methodology, rationale, and comprehensive analysis, please refer to `reports/technical_report.md`**
