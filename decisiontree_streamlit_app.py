import streamlit as st
import joblib
import numpy as np

# load model and encoder
model = joblib.load("decisiontree_mymodel.pkl")
le = joblib.load("label_encoder.pkl")

st.title("Loan Approval Prediction App")

st.write("Enter customer details 👇")

# inputs
age = st.number_input("Age", min_value=18, max_value=70)
income = st.number_input("Income", min_value=10000, step=1000)

student = st.selectbox("Student", ["Yes", "No"])
credit_score = st.number_input("Credit Score", min_value=650, max_value=900)

# encode Student
student_val = 1 if student == "Yes" else 0

if st.button("Predict Loan Status"):
    input_data = np.array([[age, income, student_val, credit_score]])
    
    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)

    if result[0] == "Yes":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
