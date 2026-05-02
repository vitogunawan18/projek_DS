import streamlit as st
import pandas as pd
import joblib

# Load model dan encoders
@st.cache_resource
def load_model():
    data = joblib.load('model/model.pkl')
    return data['model'], data['encoders'], data['features']

model, encoders, features = load_model()

st.title("Prediksi Attrition Karyawan")
st.write("Masukkan data karyawan untuk memprediksi kemungkinan *resign* (Attrition).")

# Membuat form input dinamis berdasarkan features
input_data = {}
st.sidebar.header("Input Data Karyawan")

# Kita asumsikan input default untuk demo
default_inputs = {
    'Age': 30, 'DailyRate': 800, 'DistanceFromHome': 10, 'Education': 3, 
    'EnvironmentSatisfaction': 3, 'HourlyRate': 60, 'JobInvolvement': 3, 
    'JobLevel': 2, 'JobSatisfaction': 3, 'MonthlyIncome': 5000, 
    'MonthlyRate': 15000, 'NumCompaniesWorked': 2, 'PercentSalaryHike': 15, 
    'PerformanceRating': 3, 'RelationshipSatisfaction': 3, 'StockOptionLevel': 1, 
    'TotalWorkingYears': 10, 'TrainingTimesLastYear': 3, 'WorkLifeBalance': 3, 
    'YearsAtCompany': 5, 'YearsInCurrentRole': 3, 'YearsSinceLastPromotion': 1, 
    'YearsWithCurrManager': 3, 'WorkingAgeRatio': 10/30
}

# Categorical defaults
cat_defaults = {
    'BusinessTravel': 'Travel_Rarely', 'Department': 'Sales', 
    'EducationField': 'Marketing', 'Gender': 'Male', 
    'JobRole': 'Sales Executive', 'MaritalStatus': 'Single', 
    'OverTime': 'Yes'
}

col1, col2 = st.columns(2)

with col1:
    for feat in features[:len(features)//2]:
        if feat in encoders:
            options = list(encoders[feat].classes_)
            input_data[feat] = st.selectbox(feat, options, index=options.index(cat_defaults.get(feat, options[0])))
        else:
            val = default_inputs.get(feat, 0)
            input_data[feat] = st.number_input(feat, value=float(val))

with col2:
    for feat in features[len(features)//2:]:
        if feat in encoders:
            options = list(encoders[feat].classes_)
            input_data[feat] = st.selectbox(feat, options, index=options.index(cat_defaults.get(feat, options[0])))
        else:
            val = default_inputs.get(feat, 0)
            input_data[feat] = st.number_input(feat, value=float(val))

if st.button("Prediksi Attrition"):
    df_input = pd.DataFrame([input_data])
    
    # Feature Engineering yang sama dengan notebook
    if 'Age' in df_input.columns and 'TotalWorkingYears' in df_input.columns:
        df_input['WorkingAgeRatio'] = df_input['TotalWorkingYears'] / df_input['Age']
    
    # Encoding
    for col in encoders:
        if col in df_input.columns:
            df_input[col] = encoders[col].transform(df_input[col])
            
    # Reorder columns to match training features
    df_input = df_input[features]
    
    prediction = model.predict(df_input)
    prob = model.predict_proba(df_input)[0][1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ Karyawan Berisiko Tinggi untuk Resign! (Probabilitas: {prob:.2%})")
    else:
        st.success(f"✅ Karyawan Kemungkinan Besar Akan Bertahan. (Probabilitas Resign: {prob:.2%})")
