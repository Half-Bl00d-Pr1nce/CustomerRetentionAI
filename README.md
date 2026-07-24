## Live Demo

🚀 Streamlit App: https://customerretentionai-e8wwatr7fcub9rtpzxyjqc.streamlit.app/

# 📉 Customer Churn Prediction & Retention Analytics

An end-to-end Machine Learning project that predicts telecom customer churn using the IBM Telco Customer Churn dataset. The project covers the complete machine learning lifecycle, including data preprocessing, feature engineering, model development, imbalanced learning, explainability, business ROI analysis, and deployment using Streamlit.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers is significantly more expensive than retaining them, making early identification of customers likely to churn a valuable business capability.

This project develops a machine learning solution that predicts whether a customer is likely to churn and provides actionable recommendations to support customer retention strategies.

---

## 🎯 Objectives

- Predict customer churn using machine learning.
- Handle class imbalance using multiple sampling strategies.
- Compare baseline, cost-sensitive, and advanced machine learning models.
- Interpret model predictions using SHAP explainability.
- Evaluate the financial impact of the model through ROI analysis.
- Deploy the final model as an interactive Streamlit web application.

---

## 📂 Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains customer demographic information, subscribed services, billing details, and churn labels.

Target Variable:

- Churn
  - Yes
  - No

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)
- SHAP
- Joblib
- Streamlit

---

## 📁 Repository Structure

```text
CustomerRetentionAI/

├── app/
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Baseline_Model.ipynb
│   ├── 05_Imbalanced_Learning.ipynb
│   ├── 06_Model_Optimization.ipynb
│   ├── 07_Cost_Sensitive_Learning.ipynb
│   ├── 08_Advanced_Models.ipynb
│   ├── 09_Hyperparameter_Tuning.ipynb
│   ├── 10_SHAP_Explainability.ipynb
│   ├── 11_Model_Interpretation_and_Error_Analysis.ipynb
│   ├── 12_Business_ROI.ipynb
│   └── 13_Streamlit_Deployment.ipynb
│
├── src/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

## 🔄 Project Workflow

1. Exploratory Data Analysis
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Baseline Model Development
5. Imbalanced Learning
6. Cost-Sensitive Learning
7. Advanced Model Comparison
8. Hyperparameter Tuning
9. SHAP Explainability
10. Error Analysis
11. Business ROI Analysis
12. Streamlit Deployment

---

## ⚙️ Machine Learning Pipeline

- Data Cleaning
- Feature Engineering
- Logistic Regression Baseline
- Random Forest Baseline
- SMOTE
- Cost-Sensitive Learning
- XGBoost
- LightGBM
- CatBoost
- Hyperparameter Tuning
- SHAP Explainability
- Business ROI Analysis
- Streamlit Deployment

---

## 📊 Model Performance

| Model | F1 Score | ROC-AUC |
|--------|---------:|---------:|
| Logistic Regression | 0.579 | 0.835 |
| SMOTE Logistic Regression | **0.612** | **0.832** |
| Balanced Logistic Regression | 0.608 | 0.834 |
| Random Forest | 0.538 | 0.809 |
| XGBoost | 0.543 | 0.803 |
| LightGBM | 0.563 | 0.829 |
| CatBoost | 0.564 | 0.827 |

The SMOTE Logistic Regression model achieved the highest F1-score while maintaining strong ROC-AUC performance and was selected as the final deployment model.

---

## 💼 Business Impact

The project goes beyond predictive modeling by evaluating the financial value of the deployed solution.

Business analysis includes:

- Customer churn prediction
- Business ROI estimation
- Campaign cost analysis
- Revenue preservation
- Net business benefit
- Customer retention recommendations

Estimated ROI:

**404.33%**

---

## 🖥️ Streamlit Deployment

The final model has been deployed using **Streamlit**.

Features include:

- Interactive customer input form
- Automatic feature engineering
- Real-time churn prediction
- Churn probability estimation
- Risk categorization
- Business recommendations

### Application Preview

> *(Insert screenshots here after uploading them.)*

Example:

```
📷 Home Screen

[Image]

📷 Prediction Results

[Image]

📷 Business Recommendation

[Image]
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CustomerRetentionAI.git
```

Navigate into the project

```bash
cd CustomerRetentionAI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## 🔮 Future Improvements

- Cloud deployment using Streamlit Community Cloud or AWS
- Batch prediction using uploaded CSV files
- SHAP explanations for individual predictions
- REST API integration
- Live customer database integration
- Dashboard for monitoring churn trends

---

## 👨‍💻 Author

**Sanjay Siddarth S**

B.Tech – Naval Architecture & Ocean Engineering

Indian Institute of Technology Madras

GitHub: *(Add your GitHub profile link here)*

LinkedIn: *(Add your LinkedIn profile link here)*

---

## ⭐ If you found this project useful, consider giving it a star!