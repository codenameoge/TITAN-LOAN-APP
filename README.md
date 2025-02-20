# NCAIR PILOT Loan Approval App

An AI-powered loan approval web application built using Streamlit, Scikit-learn, and SHAP for model interpretability.

## 📋 Overview
The **NCAIR PILOT Loan App** empowers financial decisions by leveraging AI to predict loan approval outcomes. This application allows users to input loan application details and receive a prediction on whether their loan is approved or rejected. It also provides detailed explanations of the factors influencing the decision using SHAP values.

## 🛠️ Features
- **Loan Approval Prediction:** Predict whether a loan will be approved or rejected.
- **User Input:** Customizable input fields for loan applicant data.
- **SHAP Model Interpretability:** Provides transparent reasons for loan approval or rejection.
- **Interactive UI:** Expandable section with a breakdown of positive and negative factors affecting the decision.
- **Scalability:** Supports large datasets with pre-processing capabilities.

## 📂 Project Structure
```
loan_approval_app/
├── Loan.csv                   # Dataset for model training and feature analysis
├── loan_approval_model.pkl    # Trained Random Forest model
├── Scaler.pkl                 # Pre-trained scaler for input normalization
└── app.py                     # Main Streamlit application
```

## 🧰 Requirements
Ensure you have the following dependencies installed:

```bash
Python >= 3.8
streamlit
scikit-learn
shap
numpy
pandas
joblib
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/your-repo/loan_approval_app.git
cd loan_approval_app
```

2. Ensure the required models and datasets are in place:
   - `loan_approval_model.pkl`: Pre-trained loan approval model
   - `Scaler.pkl`: Scaler for normalizing input data
   - `Loan.csv`: Dataset used for SHAP explanations

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open the app in your browser at `http://localhost:8501`.

## 📊 Model Input Features
| Feature                  | Description                            | Type    |
|--------------------------|----------------------------------------|---------|
| AnnualIncome             | Annual Income in USD                   | Numeric |
| EducationLevel           | Highest level of education achieved    | Categorical |
| TotalAssets              | Total financial assets in USD          | Numeric |
| Age                      | Applicant's age                        | Numeric |
| Experience               | Years of work experience               | Numeric |
| EmploymentStatus         | Employment status (Employed, etc.)     | Categorical |
| HomeOwnershipStatus      | Type of home ownership                 | Categorical |
| TotalLiabilities         | Total expenses in USD                  | Numeric |
| PreviousLoanDefaults     | Previous loan defaults (Yes/No)        | Binary  |
| BankruptcyHistory        | History of bankruptcy (Yes/No)         | Binary  |
| LoanAmount               | Loan amount requested in USD           | Numeric |
| LoanDuration             | Duration of loan in months             | Numeric |
| MonthlyLoanPayment       | Monthly loan payment in USD            | Numeric |
| TotalDebtToIncomeRatio   | Ratio of debt to income (0.0 to 1.0)   | Numeric |

## 📘 How It Works

1. **User Input:**
   - Users enter loan application details in the sidebar.
   - Categorical fields are mapped to numeric values using defined mappings.

2. **Data Pre-processing:**
   - Input data is scaled using a pre-loaded scaler (`Scaler.pkl`).

3. **Prediction:**
   - The Random Forest model (`loan_approval_model.pkl`) predicts loan approval status and probability.

4. **Interpretability:**
   - SHAP values are calculated to explain which features influenced the model's decision.

5. **Output:**
   - Displays whether the loan is approved or rejected.
   - Shows a detailed explanation of positive and negative contributing factors.

## 📊 SHAP Interpretation
- **Positive Impact:** Factors that increase the likelihood of loan approval.
- **Negative Impact:** Factors that reduce the likelihood of loan approval.

## 🔍 Customization
You can extend or modify the application by:
- Updating the model (`loan_approval_model.pkl`) with a new trained classifier.
- Adding new features by extending the `features` dictionary and updating mappings.
- Adjusting SHAP explanations for better user comprehension.

## 🧹 Troubleshooting
1. **Missing Models or Data:** Ensure `loan_approval_model.pkl`, `Scaler.pkl`, and `Loan.csv` are in the working directory.

2. **Feature Mismatch:** Ensure input data and model features are consistent.

3. **SHAP Calculation Issues:** Ensure the dataset used for SHAP aligns with the model's input features.

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contribution
Feel free to open issues and submit pull requests to enhance the application.

## 📧 Contact
For inquiries or support, please contact [ezedozien@gmail.com].

