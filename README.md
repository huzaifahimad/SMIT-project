# 📡 Telco Customer Churn Prediction & Comparative ML Analysis Dashboard

> **🔗 Live App:** [Launch Dashboard on Streamlit Cloud](https://smit-project-bbhuzzaifahimad.streamlit.app/)

---

## 🎯 Business Problem

Customer churn — the rate at which subscribers discontinue their service — is one of the most critical KPIs in the telecommunications industry. **Acquiring a new customer costs 5–7× more** than retaining an existing one. Every churned customer directly impacts recurring monthly revenue and long-term profitability.

This project delivers a **production-grade, end-to-end machine learning pipeline** that proactively identifies high-risk customers before they leave. By mapping churn-prone profiles, marketing teams and customer success managers can deploy **precision retention interventions** — targeted loyalty incentives, personalized service plans, or contract renegotiations — to stabilize revenue.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Raw Records** | 7,043 rows |
| **After Cleaning** | 7,032 rows (11 whitespace nulls in TotalCharges dropped) |
| **Features** | 21 columns — demographics, services, billing, account |
| **Target Variable** | `Churn` (Yes / No) |
| **Class Balance** | 73.5% No Churn · 26.5% Churn (moderate imbalance) |

### Feature Groups

| Category | Features |
|----------|----------|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Account & Billing** | Contract, PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges |

---

## 🛠️ Technology Stack

| Layer | Tools & Libraries |
|-------|-------------------|
| **Data Processing** | Pandas 2.2, NumPy 1.26 |
| **Visualization** | Matplotlib 3.8, Seaborn 0.13, Plotly 5.19 |
| **Machine Learning** | Scikit-learn 1.4, XGBoost 2.0 |
| **Hyperparameter Tuning** | GridSearchCV (cross-validated, scoring=F1) |
| **Imbalance Handling** | imbalanced-learn 0.12 |
| **Model Serialization** | Joblib 1.3 (.pkl pipelines) |
| **Dashboard** | Streamlit 1.32 |
| **Deployment** | Streamlit Community Cloud |

---

## 🤖 Model Evaluation Leaderboard

All 6 mandatory models + 2 hyperparameter-tuned variants, evaluated on a stratified 80/20 test split (1,407 records):

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🏆 1 | **XGBoost (Tuned)** | 0.7946 | 0.6316 | 0.5455 | 0.5854 | **0.8381** |
| 2 | Logistic Regression | 0.8038 | 0.6494 | 0.5695 | 0.6068 | 0.8362 |
| 3 | RF (Tuned) | 0.7960 | 0.6436 | 0.5214 | 0.5761 | 0.8347 |
| 4 | Random Forest | 0.7918 | 0.6364 | 0.5053 | 0.5633 | 0.8213 |
| 5 | XGBoost (Baseline) | 0.7747 | 0.5846 | 0.5267 | 0.5541 | 0.8119 |
| 6 | KNN | 0.7619 | 0.5512 | 0.5615 | 0.5563 | 0.7961 |
| 7 | SVM | 0.7960 | 0.6505 | 0.5027 | 0.5671 | 0.7872 |
| 8 | Decision Tree | 0.7569 | 0.5426 | 0.5455 | 0.5440 | 0.7840 |

> **Champion Model:** XGBoost (Tuned) — selected for highest ROC-AUC (0.838), providing the best class separation across all probability thresholds. Tuned via GridSearchCV with `n_estimators`, `max_depth`, and `learning_rate` optimization.

---

## 🔑 Key EDA Findings

1. **Contract Type is the #1 churn driver** — Month-to-month contracts show a **42.7% churn rate** vs. 11.3% (One-year) and just 2.8% (Two-year). Customers without commitment churn at 15× the rate of long-term contracts.

2. **Fiber Optic users churn at 2× the rate of DSL** — 41.9% vs. 19.0%. This likely reflects pricing dissatisfaction or service quality issues rather than the technology itself.

3. **Electronic check payment = highest churn** — 45.3% churn rate vs. ~15-16% for auto-pay methods. Manual payment friction correlates strongly with disengagement.

4. **Short-tenure customers are highest risk** — Customers with < 12 months tenure are disproportionately represented in churn. Long-tenure customers (60+ months) almost never churn.

5. **Senior Citizens churn at nearly 2× the rate** — 41.7% vs. 23.6% for non-seniors, suggesting age-specific service gaps.

6. **Missing security/support services amplify churn** — Customers without Online Security or Tech Support show significantly elevated churn probability.

---

## 🖥️ Dashboard Sections (Streamlit)

The interactive web application includes **6 fully functional sections** as required:

| # | Section | Description |
|---|---------|-------------|
| 1 | **🏠 Home** | Business context, project pipeline overview, dataset feature groups |
| 2 | **📊 Dataset Explorer** | Filterable data grid, `df.describe()` statistics, column profiles, data quality report (nulls, duplicates, dtypes) |
| 3 | **🔍 EDA Dashboard** | Interactive Plotly charts with dropdown selectors — univariate (histogram/box/violin), bivariate (12 features vs churn), correlation heatmap, target analysis |
| 4 | **🤖 Model Training** | Select any of 6 models, configure hyperparameters via sliders, train live, view instant metrics + confusion matrix + ROC curve |
| 5 | **🏆 Model Comparison** | Performance leaderboard, Accuracy vs F1 bars, multi-model ROC curves, confusion matrix grid, Precision-Recall curves, feature importance |
| 6 | **🎯 Prediction System** | Customer input form (demographics, services, billing) → serialized .pkl pipeline → **Churn Risk / Retained Account** result with confidence score probability |

---

## 🚀 Local Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/huzaifahimad/SMIT-project.git
cd SMIT-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📁 Repository Structure

```
SMIT-project/
│
├── data/                  # Raw CSV dataset
│   └── Telco-Customer-Churn.csv
│
├── notebooks/             # Jupyter Notebook — full EDA, modeling, tuning
│   └── Telco_Churn_Complete_Analysis.ipynb
│
├── models/                # Serialized production pipeline binaries
│   ├── champion_model.pkl        # XGBoost Tuned (production)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── random_forest_tuned.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl                # StandardScaler for feature scaling
│   ├── feature_columns.pkl       # Feature column order
│   └── metrics.json              # All 8 model metrics
│
├── visuals/               # Exported EDA and comparison charts
│   ├── 01_churn_distribution.png
│   ├── 02_univariate_numerical.png
│   ├── 03_demographics_churn.png
│   ├── 04_bivariate_service_churn.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_model_comparison.png
│   ├── 07_roc_curves.png
│   ├── 08_confusion_matrices.png
│   ├── 09_precision_recall_curves.png
│   └── 10_feature_importance.png
│
├── app.py                 # Streamlit dashboard (main entry point)
├── utils.py               # Data loading, preprocessing, model utilities
├── requirements.txt       # Pinned dependency versions
├── .gitignore             # Python/IDE/OS exclusions
└── README.md              # This documentation
```

---

## 📝 Project Phases Completed

| Phase | Deliverable | Status |
|-------|-------------|--------|
| Phase 1 | Problem understanding & business goal definition | ✅ |
| Phase 2 | Dataset profiling — 7,032 records, hidden nulls discovered | ✅ |
| Phase 3 | EDA — 10 visualizations with documented insights | ✅ |
| Phase 4 | Preprocessing — encoding, scaling, stratified split, no leakage | ✅ |
| Phase 5 | 6 mandatory ML models trained & evaluated | ✅ |
| Phase 6 | Model comparison — 5 required visualization types | ✅ |
| Phase 7 | Hyperparameter tuning — GridSearchCV for RF & XGBoost | ✅ |
| Phase 8 | Streamlit dashboard — all 6 interactive sections | ✅ |
| Phase 9 | Cloud deployment on Streamlit Community Cloud | ✅ |
| Phase 10 | GitHub repository with professional structure | ✅ |

---

**Built with ❤️ using Python, Scikit-learn, XGBoost & Streamlit**
