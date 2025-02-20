import streamlit as st
import shap
import numpy as np
import pandas as pd
import joblib  # For loading saved models
from sklearn.ensemble import RandomForestClassifier  # Example model

# Load the trained model
model = joblib.load("loan_approval_model.pkl")

# Load dataset
df = pd.read_csv("Loan.csv")  
X = ['AnnualIncome', 'EducationLevel', 'TotalAssets', 'Age', 'Experience', 'EmploymentStatus', 'HomeOwnershipStatus', 'TotalLiabilities', 'PreviousLoanDefaults', 'BankruptcyHistory', 'LoanDuration', 'MonthlyLoanPayment', 'LoanAmount', 'TotalDebtToIncomeRatio']

employment_status_mapping = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
educationLevel_mapping = {'High School': 0, 'Bachelor': 2, 'Master': 3, 'Associate': 1, 'Doctorate': 4}
homeOwnershipStatus_mapping = {'Own': 0, 'Mortgage': 1, 'Rent': 2, 'Other': 3}

df['EmploymentStatus'] = df['EmploymentStatus'].map(employment_status_mapping)
df['EducationLevel'] = df['EducationLevel'].map(educationLevel_mapping)
df['HomeOwnershipStatus'] = df['HomeOwnershipStatus'].map(homeOwnershipStatus_mapping)

X = df[X] 

st.set_page_config(page_title="Loan App", page_icon="💳", layout="centered")

st.markdown("""
    <style>
    .decision-container {border: 2px solid #007bff; border-radius: 10px; padding: 15px; margin-top: 20px;}
    .positive {color: #28a745;}
    .negative {color: #dc3545;}
    .clickable-title {color: #007bff; cursor: pointer; font-size: 20px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.title("💳  NCAIR PILOT Loan App")
st.subheader("Empowering financial decisions with AI-driven insights.")


st.sidebar.header("Enter Loan Application Details")
features = {}

# Customizing input fields
features['AnnualIncome'] = st.sidebar.number_input("Annual Income ($)", min_value=0, step=1000)
features['EducationLevel'] = educationLevel_mapping[st.sidebar.selectbox("Education Level", options=list(educationLevel_mapping.keys()))]
features['TotalAssets'] = st.sidebar.number_input("Total Assets ($)", min_value=0, step=10000)
features['Age'] = st.sidebar.number_input("Age", min_value=18, max_value=90, step=1)
features['Experience'] = st.sidebar.number_input("Years of Work Experience", min_value=0, max_value=60, step=1)
features['EmploymentStatus'] = employment_status_mapping[st.sidebar.selectbox("Employment Status", options=list(employment_status_mapping.keys()))]
features['HomeOwnershipStatus'] = homeOwnershipStatus_mapping[st.sidebar.selectbox("Home Ownership Status", options=list(homeOwnershipStatus_mapping.keys()))]
features['TotalLiabilities'] = st.sidebar.number_input("Total Expenses ($)", min_value=0, step=5000)
features['TotalDebtToIncomeRatio'] = st.sidebar.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, step=0.01)
features['PreviousLoanDefaults'] = 1 if st.sidebar.selectbox("Previous Loan Defaults", options=["No", "Yes"], index=0) == "Yes" else 0
features['BankruptcyHistory'] = 1 if st.sidebar.selectbox("Bankruptcy History", options=["No", "Yes"], index=0) == "Yes" else 0
features['LoanAmount'] = st.sidebar.number_input("Loan Amount ($)", min_value=1000, step=5000)
features['LoanDuration'] = st.sidebar.slider("Loan Duration (Months)", min_value=12, max_value=120, step=12)
features['MonthlyLoanPayment'] = st.sidebar.number_input("Monthly Loan Payment ($)", min_value=0, step=500)

input_data = pd.DataFrame([features])

# Apply scaling
scaler = joblib.load("Scaler.pkl")
input_data[X.columns] = scaler.transform(input_data[X.columns])


# Ensure input_data has all required columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)
if st.sidebar.button("Predict Loan Approval"):
    prediction_proba = model.predict_proba(input_data)[0]  # Get probabilities
    prediction = model.predict(input_data)[0]

    print(f"Prediction Probabilities: {prediction_proba}")  # Debugging output
    print(f"Predicted Class: {prediction}")  # Debugging output


    if prediction == 1:
        st.success("✅ Loan Approved!")
        decision_message = f"Your loan was approved based on strong financial indicators."
    else:
        st.error("❌ Loan Rejected!")
        decision_message = "Your loan was declined because some financial criteria did not meet our standards."
    
    st.markdown(f'<div class="decision-container"><p class="clickable-title">🔍 Why This Decision Was Made</p><p>{decision_message}</p></div>', unsafe_allow_html=True)


    with st.expander("Click to see more"):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(input_data)
        shap_values_array = np.array(shap_values.values)
        shap_values_array = shap_values_array[0, :, prediction]
        shap_values_array = shap_values_array.flatten()

        # Ensure feature count matches
        if len(X.columns) != len(shap_values_array):
            raise ValueError(f"Feature count mismatch! Expected {len(X.columns)}, got {len(shap_values_array)}.")

        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "SHAP Value": shap_values_array
        })

        # Separate positive and negative impacts
        positive_impact = feature_importance[feature_importance["SHAP Value"] > 0].sort_values(by="SHAP Value", ascending=False)
        negative_impact = feature_importance[feature_importance["SHAP Value"] < 0].sort_values(by="SHAP Value", ascending=False)

        # Map features to user-friendly explanations
        positive_explanations = {
        "AnnualIncome": "have a high annual income",
        "EducationLevel": "have a strong educational background",
        "TotalAssets": "have a solid financial portfolio",
        "Age": "are a suitable age for financial stability",
        "Experience": "have significant work experience",
        "EmploymentStatus": "have a stable employment",
        "HomeOwnershipStatus": "have your own home",
        "TotalLiabilities": "have minimal expenses",
        "PreviousLoanDefaults": "have a good track record with no loan defaults",
        "BankruptcyHistory": "have no history of bankruptcy",
        "LoanDuration": "chose a manageable loan duration",
        "MonthlyLoanPayment": "set an affordable monthly loan payment",
        "LoanAmount": "requested for a reasonable loan amount",
        "TotalDebtToIncomeRatio": "maintain a low debt profile"
    }
        
        negative_explanations = {
        "AnnualIncome": "have a low annual income",
        "EducationLevel": "have a weak educational background",
        "TotalAssets": "don't have substantial wealth",
        "Age": "are an unsuitable age for financial stability",
        "Experience": "have insufficient work experience",
        "EmploymentStatus": "have an unstable/inconsistent employment",
        "HomeOwnershipStatus": "don't own a home",
        "TotalLiabilities": "have a lot of expenses",
        "PreviousLoanDefaults": "have a history of previous loan defaults",
        "BankruptcyHistory": "have a history of bankruptcy",
        "LoanDuration": "chose a loan term that's difficult to manage",
        "MonthlyLoanPayment": "set a monthly loan payment you can't afford",
        "LoanAmount": "requested for a loan that exceeds your financial capacity",
        "TotalDebtToIncomeRatio": "owe more than you can comfortably afford"
    }


        # Provide simple explanations based on prediction
        if prediction == 1:
            st.write("The loan was approved because of the following factors:")
            for i in range(min(5, len(positive_impact))):
                feature_name = positive_impact.iloc[i]["Feature"]
                explanation = positive_explanations.get(feature_name, feature_name)
                st.write(f"✅ You {explanation}.")
        else:
            st.write("The loan was rejected due to the following reasons:")
            for i in range(min(5, len(negative_impact))):
                feature_name = negative_impact.iloc[i]["Feature"]
                explanation = negative_explanations.get(feature_name, feature_name)
                st.write(f"❌ You {explanation}.")


