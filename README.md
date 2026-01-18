📉 Customer Churn Prediction System
📌 Project Overview

Customer churn prediction aims to identify customers who are likely to stop using a company’s services.
This project uses machine learning to predict churn based on customer demographics, account information, and service usage patterns, helping businesses take proactive retention actions.

🎯 Problem Statement

Customer retention is critical for subscription-based businesses. Acquiring new customers is significantly more expensive than retaining existing ones.
The objective of this project is to:

Predict whether a customer will churn (leave) or not churn

Identify key factors influencing churn

Optimize the model for high churn recall, which is more important than raw accuracy

🧠 Solution Approach

The project follows a modular, industry-style ML pipeline, separating each stage into dedicated notebooks:

Exploratory Data Analysis (EDA)

Feature Engineering

Model Training

Model Evaluation

Model Tuning & Optimization

🗂️ Project Structure
Customer_Churn_Prediction/
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   ├── cleaned_data.csv
│   └── processed_data.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   └── 05_Model_Tuning.ipynb
│
├── model/
│   ├── customer_churn_model.pkl
│   └── customer_churn_model_tuned.pkl
│
├── requirements.txt
└── README.md

📊 Dataset

Source: Kaggle – Telco Customer Churn Dataset

Target Variable: Churn (0 = No, 1 = Yes)

Features include:

Demographics (gender, dependents, etc.)

Account details (contract type, tenure)

Billing & payment information

Service usage details

🔍 Exploratory Data Analysis (EDA)

Analyzed churn distribution and class imbalance

Visualized numerical features (tenure, MonthlyCharges, TotalCharges)

Studied categorical features vs churn

Identified key patterns influencing customer churn

🛠 Feature Engineering

Removed non-informative identifier (customerID)

One-hot encoded categorical features

Scaled numerical features using StandardScaler

Saved processed dataset for modeling

🤖 Model Training

Trained baseline models:

Logistic Regression

Random Forest Classifier

Used stratified train-test split

Selected Random Forest as primary model due to better performance

📈 Model Evaluation

Evaluation focused on business-relevant metrics, not just accuracy.

Key Metrics:

Precision

Recall (especially for churn class)

F1-score

ROC-AUC

Confusion Matrix

Accuracy alone was not used as the main metric due to class imbalance.

⚖️ Handling Class Imbalance

To improve churn detection:

Applied class weighting

Optimized for recall, not accuracy

Accepted a slight drop in accuracy for better churn capture

🔧 Model Tuning

Used RandomizedSearchCV

Optimized hyperparameters with recall as the scoring metric

Applied threshold tuning to further improve churn recall

Final tuned model saved for deployment

✅ Final Results (Summary)

Improved churn recall significantly compared to baseline

Model aligned with real-world business needs

Feature importance provided actionable insights

🧪 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Imbalanced-learn

Joblib

🚀 How to Run the Project

Clone the repository:

git clone <repo-link>


Install dependencies:

pip install -r requirements.txt


Run notebooks in order:

01_EDA.ipynb

02_Feature_Engineering.ipynb

03_Model_Training.ipynb

04_Model_Evaluation.ipynb

05_Model_Tuning.ipynb

📌 Key Learnings

Accuracy is misleading for imbalanced problems

Recall is critical for churn prediction

Proper feature consistency is essential

Business context matters more than raw metrics

📬 Contact

Author: Krish Gaur
Domain: Data Science / Machine Learning
Use Case: Customer Analytics, Retention Strategy

If you want, next I can:

🔥 Rewrite this README in shorter recruiter-friendly version

🌐 Create a Streamlit app

🧾 Write resume bullet points for this project

📊 Create before vs after metric table

Just tell me what you want next.
