# Dataset Source and Attribution

## Dataset Information

**File:** `part_3_customer_churn_prediction.csv`

**Description:**
This dataset contains customer information for a fictional telecom company, including demographic data, service usage, account information, and churn status. It is designed for educational and analytical purposes in customer churn prediction.

## Dataset Characteristics

- **Total Records:** 1,800 customers
- **Features:** 20 independent variables + 1 target variable (Churn)
- **Target Variable:** Binary classification (Yes/No)
- **Class Balance:** Moderately imbalanced (64.1% No, 35.9% Yes)
- **Missing Data:** Minimal (0.6% in TotalCharges column)

## Feature Categories

### Demographic (4 features)
- Gender, SeniorCitizen, Partner, Dependents

### Service Usage (10 features)
- PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup
- DeviceProtection, TechSupport, StreamingTV, StreamingMovies

### Account Information (5 features)
- Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

### Customer Lifecycle (1 feature)
- Tenure (months with company)

## Data Quality Notes

1. **Missing Values:** 11 records have missing TotalCharges. These primarily correspond to customers with tenure=0 (new customers who haven't been billed yet).
2. **Special Values:** Service columns contain "No internet service" and "No phone service" values that are semantically equivalent to "No" for customers without those services.
3. **Data Types:** TotalCharges is stored as object type due to missing values and requires conversion to numeric after imputation.

## Usage Terms

This dataset is provided for educational and analytical purposes. It is a synthetic dataset created for learning machine learning concepts, specifically:
- Exploratory Data Analysis (EDA)
- Data Preprocessing and Feature Engineering
- Classification Model Building
- Model Evaluation and Interpretation
- Business Strategy Development

## Recommended Analysis Pipeline

1. Data Loading and Validation
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Preprocessing
4. Feature Engineering
5. Model Training (Multiple Algorithms)
6. Model Evaluation and Selection
7. Feature Importance Analysis
8. Business Insight Generation
9. Retention Strategy Formulation

## Citation

If using this dataset for academic or research purposes, please cite:
```
Telecom Customer Churn Dataset (Educational Version)
Synthetic dataset for machine learning education in customer analytics
```
