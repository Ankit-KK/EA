import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Load the pre-trained encoder and scaler
encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('minmax_scaler.pkl')  # Assuming you've saved your scaler similarly

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Set the title of the app
st.title("Employee Attrition Prediction")

# Option to choose input method
input_method = st.radio("Choose input method", ["Single Person", "CSV File"])

if input_method == "Single Person":
    st.write("Enter the details for a single person:")

    # Input fields for a single person's data
    age = st.number_input('Age', min_value=18, max_value=100)
    daily_rate = st.number_input('DailyRate', min_value=1)
    distance_from_home = st.number_input('DistanceFromHome', min_value=0)
    education = st.selectbox('Education', options=[1, 2, 3, 4])  # Adjust options based on your data
    environment_satisfaction = st.selectbox('EnvironmentSatisfaction', options=[1, 2, 3, 4])
    hourly_rate = st.number_input('HourlyRate', min_value=1)
    job_involvement = st.selectbox('JobInvolvement', options=[1, 2, 3, 4])
    job_level = st.number_input('JobLevel', min_value=1)
    job_satisfaction = st.selectbox('JobSatisfaction', options=[1, 2, 3, 4])
    monthly_income = st.number_input('MonthlyIncome', min_value=1)
    monthly_rate = st.number_input('MonthlyRate', min_value=1)
    num_companies_worked = st.number_input('NumCompaniesWorked', min_value=0)
    over_time = st.selectbox('OverTime', options=['Yes', 'No'])
    percent_salary_hike = st.number_input('PercentSalaryHike', min_value=0)
    performance_rating = st.selectbox('PerformanceRating', options=[1, 2, 3, 4])
    relationship_satisfaction = st.selectbox('RelationshipSatisfaction', options=[1, 2, 3, 4])
    stock_option_level = st.number_input('StockOptionLevel', min_value=0)
    total_working_years = st.number_input('TotalWorkingYears', min_value=0)
    training_times_last_year = st.number_input('TrainingTimesLastYear', min_value=0)
    work_life_balance = st.selectbox('WorkLifeBalance', options=[1, 2, 3, 4])
    years_at_company = st.number_input('YearsAtCompany', min_value=0)
    years_in_current_role = st.number_input('YearsInCurrentRole', min_value=0)
    years_since_last_promotion = st.number_input('YearsSinceLastPromotion', min_value=0)
    years_with_curr_manager = st.number_input('YearsWithCurrManager', min_value=0)

    if st.button('Predict'):
        # Prepare the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'DailyRate': [daily_rate],
            'DistanceFromHome': [distance_from_home],
            'Education': [education],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'HourlyRate': [hourly_rate],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobSatisfaction': [job_satisfaction],
            'MonthlyIncome': [monthly_income],
            'MonthlyRate': [monthly_rate],
            'NumCompaniesWorked': [num_companies_worked],
            'OverTime': [over_time],
            'PercentSalaryHike': [percent_salary_hike],
            'PerformanceRating': [performance_rating],
            'RelationshipSatisfaction': [relationship_satisfaction],
            'StockOptionLevel': [stock_option_level],
            'TotalWorkingYears': [total_working_years],
            'TrainingTimesLastYear': [training_times_last_year],
            'WorkLifeBalance': [work_life_balance],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'YearsSinceLastPromotion': [years_since_last_promotion],
            'YearsWithCurrManager': [years_with_curr_manager]
        })

        # Pre-process the data
        input_data = input_data.drop(['Over18'], axis=1, errors='ignore')
        input_data['OverTime'] = input_data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)
        input_data = input_data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')

        x_cat_input = input_data.select_dtypes(include=['object'])

        # Apply OneHotEncoder
        x_cat_input = encoder.transform(x_cat_input).toarray()
        x_cat_input = pd.DataFrame(x_cat_input)

        # Select numerical columns
        x_num_input = input_data.select_dtypes(include=['float', 'int'])

        # Combine the categorical and numerical features
        x_all_input = pd.concat([x_cat_input, x_num_input], axis=1)
        x_all_input.columns = x_all_input.columns.astype(str)

        # Scale the features
        x_input = scaler.transform(x_all_input)

        # Make prediction
        y_pred_input = model.predict(x_input)
        y_pred_input = (y_pred_input > 0.5).astype(int)
        prediction = pd.DataFrame(y_pred_input, columns=['Attrition Prediction'])

        st.write("Prediction for the entered data:")
        st.write(prediction)

elif input_method == "CSV File":
    st.write("Upload a CSV file to predict employee attrition.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        new_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(new_data.head())  # Display the first few rows of the uploaded data

        # Pre-process the data
        new_data = new_data.drop('Over18', axis=1, errors='ignore')
        new_data['OverTime'] = new_data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)
        new_data = new_data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1, errors='ignore')

        # Separate categorical and numerical data
        x_cat_new = new_data.select_dtypes(include=['object'])

        # Apply OneHotEncoder
        x_cat_new = encoder.transform(x_cat_new).toarray()
        x_cat_new = pd.DataFrame(x_cat_new)

        # Select numerical columns
        x_num_new = new_data.select_dtypes(include=['float', 'int'])

        # Combine the categorical and numerical features
        x_all_new = pd.concat([x_cat_new, x_num_new], axis=1)
        x_all_new.columns = x_all_new.columns.astype(str)

        # Scale the features
        x_new = scaler.transform(x_all_new)

        # Make predictions
        y_pred_new = model.predict(x_new)
        y_pred_new = (y_pred_new > 0.5).astype(int)

        # Display the predictions
        predictions = pd.DataFrame(y_pred_new, columns=['Attrition Prediction'])
        st.write("Predictions:")
        st.write(predictions)

        # Option to download the predictions as a CSV file
        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
    else:
        st.write("Please upload a CSV file to continue.")
