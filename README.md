# Customer Churn Prediction for Telecom Company

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **End-to-end machine learning project predicting customer churn for a telecom company using classification algorithms, with actionable business retention strategies.**

---

## Business Problem

Customer churn is one of the most critical challenges facing the telecom industry. Acquiring a new customer costs **5-25x more** than retaining an existing one. This project analyzes 1,800 customer records to:

1. **Identify** the key drivers of customer churn
2. **Predict** which customers are at risk of leaving
3. **Recommend** data-driven retention strategies to reduce churn and protect revenue

**Current State:** The company has a **35.9% churn rate** — significantly above the telecom industry average of 15-25%, indicating a serious retention problem requiring immediate intervention.

---

## Dataset Description

| Attribute | Type | Description |
|-----------|------|-------------|
| `CustomerID` | String | Unique customer identifier |
| `Gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary | 1 = Senior citizen (65+), 0 = Not |
| `Partner` | Binary | Has partner (Yes/No) |
| `Dependents` | Binary | Has dependents (Yes/No) |
| `Tenure` | Numerical | Months with the company (0-72) |
| `PhoneService` | Binary | Has phone service (Yes/No) |
| `MultipleLines` | Categorical | Multiple phone lines |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Binary | Has online security (Yes/No) |
| `OnlineBackup` | Binary | Has online backup (Yes/No) |
| `DeviceProtection` | Binary | Has device protection (Yes/No) |
| `TechSupport` | Binary | Has tech support (Yes/No) |
| `StreamingTV` | Binary | Has streaming TV (Yes/No) |
| `StreamingMovies` | Binary | Has streaming movies (Yes/No) |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Binary | Uses paperless billing (Yes/No) |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numerical | Monthly bill amount ($) |
| `TotalCharges` | Numerical | Total amount charged to date ($) |
| **Churn** | **Target** | **Customer left (Yes/No)** |

**Dataset Size:** 1,800 rows × 21 columns  
**Class Distribution:** 64.1% Retained | 35.9% Churned  
**Missing Values:** 11 in `TotalCharges` (0.6%)

---

## Data Cleaning and Preprocessing Summary

| Step | Action | Rationale |
|------|--------|-----------|
| **1. Missing Values** | Imputed `TotalCharges` using `MonthlyCharges × Tenure` | Logical relationship for subscription billing |
| **2. Remove ID** | Dropped `CustomerID` | Unique identifier, not predictive |
| **3. Binary Encoding** | Mapped Yes/No → 1/0 for all binary columns | Required for ML algorithms |
| **4. Service Consolidation** | Replaced "No internet service" / "No phone service" with "No" | Eliminates redundant category |
| **5. One-Hot Encoding** | Encoded `InternetService`, `Contract`, `PaymentMethod` | Converts categorical to numerical |
| **6. Feature Scaling** | Standardized `Tenure`, `MonthlyCharges`, `TotalCharges` | Ensures equal feature contribution |
| **7. Train-Test Split** | 80/20 stratified split | Maintains class distribution |

**Final Feature Count:** 24 features after encoding  
**Training Set:** 1,440 samples | **Test Set:** 360 samples

---

## EDA Insights

### Overall Churn Rate
![Overall Churn](images/01_overall_churn_rate.png)

> **35.9% churn rate** — roughly 1 in 3 customers leave. This is significantly above the telecom industry average, indicating a serious retention problem.

### Churn by Contract Type
![Churn by Contract](images/02_churn_by_contract.png)

> **Month-to-month customers have 42.7% churn** vs only 2.8% for two-year contracts. Longer contracts create switching costs and commitment.

### Churn by Tenure
![Churn by Tenure](images/03_churn_by_tenure.png)

> **New customers (0-12 months) have 47.7% churn** — the highest of any group. Churn drops dramatically after the first year. The "honeymoon period" is critical.

### Churn by Monthly Charges
![Churn by Charges](images/04_churn_by_monthly_charges.png)

> Higher monthly charges correlate with higher churn. Customers paying **$111+ have 49.4% churn** vs only 20.2% for low-cost plans.

### Churn by Payment Method
![Churn by Payment](images/05_churn_by_payment_method.png)

> **Electronic check users have 44.8% churn** vs ~15-17% for auto-payment methods. Electronic check may indicate less financially stable or less committed customers.

### Churn by Internet Service
![Churn by Internet](images/06_churn_by_internet_service.png)

> **Fiber optic customers have 47.8% churn** — the highest of any internet type, despite paying premium prices.

### Correlation Heatmap
![Correlation Heatmap](images/08_correlation_heatmap.png)

> The strongest predictors are contract type, tenure, and total charges. Month-to-month contracts show strong positive correlation with churn.

---

## Models Used

Four classification algorithms were trained and evaluated:

| Model | Description | Why Chosen |
|-------|-------------|------------|
| **Logistic Regression** | Linear probabilistic classifier | Interpretable coefficients, baseline for comparison |
| **Decision Tree** | Tree-based rule learner | Captures non-linear relationships, easy to visualize |
| **K-Nearest Neighbors** | Instance-based learning | Simple, non-parametric, good for local patterns |
| **Support Vector Machine** | Maximum margin classifier | Effective in high-dimensional spaces, robust to overfitting |

All models were trained with `random_state=42` for reproducibility.

---

## Model Evaluation Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | **AUC-ROC** |
|-------|----------|-----------|--------|----------|-------------|
| **Logistic Regression** | **0.739** | **0.661** | 0.558 | **0.605** | **0.780** |
| Support Vector Machine | 0.744 | 0.687 | 0.527 | 0.597 | 0.768 |
| Decision Tree | 0.686 | 0.568 | 0.519 | 0.543 | 0.667 |
| K-Nearest Neighbors | 0.658 | 0.525 | 0.496 | 0.510 | 0.663 |

### ROC Curves
![ROC Curves](images/10_roc_curves.png)

> Logistic Regression (AUC=0.780) and SVM (AUC=0.768) significantly outperform Decision Tree and KNN. All models beat random chance.

### Confusion Matrix
![Confusion Matrices](images/09_confusion_matrices.png)

> All models correctly identify most retained customers but struggle more with predicting churners — common in imbalanced datasets. Logistic Regression shows the best balance.

---

## Final Model Selection

**Selected Model: Logistic Regression**

**Rationale:**
- **Highest AUC-ROC (0.780)** — best discriminative ability
- **Interpretable coefficients** — each feature's impact on churn is directly quantifiable
- **Stable performance** across all metrics
- **Fast inference** — suitable for production deployment
- **Well-calibrated probabilities** — reliable churn risk scores

### Top Predictive Features

![Feature Importance](images/11_feature_importance.png)

**Top Churn Risk Factors:**
1. `InternetService_Fiber optic` (+0.61) — Premium service dissatisfaction
2. `PaymentMethod_Electronic check` (+0.59) — Payment instability indicator
3. `TotalCharges` (+0.42) — Higher lifetime spenders more likely to leave
4. `MonthlyCharges` (+0.29) — Price sensitivity
5. `SeniorCitizen` (+0.22) — Fixed-income constraints

**Top Retention Factors:**
1. `Contract_Two year` (−1.86) — Strongest protection
2. `Contract_One year` (−1.72) — Significant commitment effect
3. `Tenure` (−0.89) — Loyalty builds over time
4. `TechSupport` (−0.37) — Support reduces frustration
5. `PaymentMethod_Credit card` (−0.22) — Auto-pay commitment

---

## Retention Strategy

### Executive Dashboard
![Executive Dashboard](images/12_executive_dashboard.png)

### Six Data-Driven Retention Initiatives

| # | Initiative | Insight | Action | Timeline | Est. Annual Impact |
|---|-----------|---------|--------|----------|-------------------|
| 1 | **Contract Conversion Campaign** | Month-to-month: 42.7% churn vs 2.8% two-year | Offer 15-20% discount for 1-year, 25-30% for 2-year | 3 months | ~$1.2M |
| 2 | **Payment Method Migration** | Electronic check: 44.8% churn vs ~15% auto-pay | $5/month discount for auto-pay enrollment | 2 months | ~$800K |
| 3 | **New Customer Onboarding** | First 12 months: 47.7% churn | 90-day "Success Journey" with proactive check-ins | 4 months | ~$480K |
| 4 | **Fiber Optic Quality Audit** | Fiber: 47.8% churn despite premium pricing | Technical infrastructure audit + price matching | 6 months | ~$1.5M |
| 5 | **Senior Retention Program** | Seniors: 38.2% churn | "Senior Advantage" plan with dedicated support | 3 months | ~$350K |
| 6 | **Predictive Intervention System** | Model achieves 78% AUC | Deploy monthly scoring with auto-triggered retention offers | 5 months | ~$720K |

**Total Estimated Annual Revenue Protection: $5.05M+**

### Implementation Roadmap

```
PHASE 1 (Months 1-2): QUICK WINS
├── Payment Method Migration
└── Senior Retention Program

PHASE 2 (Months 3-4): CORE INITIATIVES
├── Contract Conversion Campaign
└── New Customer Onboarding Program

PHASE 3 (Months 5-6): STRATEGIC INVESTMENTS
├── Fiber Optic Quality Audit
└── Predictive Intervention System Deployment
```

---

## How to Run the Project

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-prediction-project.git
cd churn-prediction-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

**Option 1: Jupyter Notebook (Recommended for exploration)**
```bash
jupyter notebook notebook.ipynb
```

**Option 2: Python Script**
```bash
python main.py
```

### Expected Output

- Console output with model metrics and business insights
- Saved visualizations in `/images/` directory
- Model performance comparison table
- Feature importance rankings
- Business recommendations with revenue impact estimates

### File Structure

```
churn-prediction-project/
├── README.md                 # This file
├── notebook.ipynb            # Complete analysis notebook
├── main.py                   # Executable Python script
├── requirements.txt          # Python dependencies
├── dataset_source.md         # Dataset attribution
├── images/                   # Generated visualizations
│   ├── 01_overall_churn_rate.png
│   ├── 02_churn_by_contract.png
│   ├── 03_churn_by_tenure.png
│   ├── 04_churn_by_monthly_charges.png
│   ├── 05_churn_by_payment_method.png
│   ├── 06_churn_by_internet_service.png
│   ├── 07_churn_by_senior_citizen.png
│   ├── 08_correlation_heatmap.png
│   ├── 09_confusion_matrices.png
│   ├── 10_roc_curves.png
│   ├── 11_feature_importance.png
│   └── 12_executive_dashboard.png
└── outputs/                  # Generated outputs
```

---

## Dependencies

See [`requirements.txt`](requirements.txt) for full dependency list. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.5.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical computing |
| scikit-learn | >=1.3.0 | Machine learning |
| matplotlib | >=3.7.0 | Visualization |
| seaborn | >=0.12.0 | Statistical visualization |
| jupyter | >=1.0.0 | Notebook environment |

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## Acknowledgments

- Dataset: Telecom customer churn dataset for educational purposes
- Analysis inspired by industry best practices in customer analytics
- Built with scikit-learn, pandas, and matplotlib

---

