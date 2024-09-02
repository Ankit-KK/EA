import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# Load the pre-trained encoder and scaler
encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('minmax_scaler.pkl')

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Set the title of the app
st.title("Employee Attrition Prediction")

# Columns to be dropped
columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']

# Option to choose input method
input_method = st.radio("Choose input method", ["Single Person", "CSV File"])

if input_method == "Single Person":
    st.subheader("Enter the details for a single person:")

    # Create a form for single person input
    with st.form(key='single_person_form'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider('Age', min_value=18, max_value=100)
            daily_rate = st.number_input('DailyRate', min_value=1)
            distance_from_home = st.number_input('DistanceFromHome', min_value=1)
            education = st.slider('Education', min_value=1, max_value=5)
            hourly_rate = st.number_input('HourlyRate', min_value=1)
            job_involvement = st.number_input('JobInvolvement', min_value=1)
            job_level = st.number_input('JobLevel', min_value=1)
            job_satisfaction = st.slider('JobSatisfaction', min_value=1, max_value=10)
            monthly_income = st.number_input('MonthlyIncome', min_value=1)
            environment_satisfaction = st.slider('EnvironmentSatisfaction', min_value=1, max_value=4)
        
        with col2:
            monthly_rate = st.number_input('MonthlyRate', min_value=1)
            num_companies_worked = st.number_input('NumCompaniesWorked', min_value=0)
            percent_salary_hike = st.number_input('PercentSalaryHike', min_value=0)
            performance_rating = st.slider('PerformanceRating', min_value=1, max_value=10)
            relationship_satisfaction = st.slider('RelationshipSatisfaction', min_value=1, max_value=10)
            stock_option_level = st.slider('StockOptionLevel', min_value=0, max_value=10)
            total_working_years = st.number_input('TotalWorkingYears', min_value=0)
            training_times_last_year = st.number_input('TrainingTimesLastYear', min_value=0)
            years_at_company = st.number_input('YearsAtCompany', min_value=0)
        
        with col3:
            business_travel = st.selectbox('BusinessTravel', options=['Travel_Rarely', 'Travel_Frequently'])
            department = st.selectbox('Department', options=['Research & Development', 'Sales'])
            education_field = st.selectbox('EducationField', options=['Life Sciences', 'Medical', 'Other'])
            gender = st.selectbox('Gender', options=['Male', 'Female'])
            job_role = st.selectbox('JobRole', options=['Laboratory Technician', 'Research Scientist', 'Manufacturing Director', 'Sales Executive', 'Healthcare Representative'])
            marital_status = st.selectbox('MaritalStatus', options=['Single', 'Married', 'Divorced'])
            over_time = st.selectbox('OverTime', options=['Yes', 'No'])
            work_life_balance = st.slider('WorkLifeBalance', min_value=1, max_value=10)
            years_in_current_role = st.number_input('YearsInCurrentRole', min_value=0)
            years_since_last_promotion = st.number_input('YearsSinceLastPromotion', min_value=0)
            years_with_curr_manager = st.number_input('YearsWithCurrManager', min_value=0)
        
        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Prepare the input data
            input_data = pd.DataFrame({
                'Age': [age],
                'BusinessTravel': [business_travel],
                'DailyRate': [daily_rate],
                'Department': [department],
                'DistanceFromHome': [distance_from_home],
                'Education': [education],
                'EducationField': [education_field],
                'EnvironmentSatisfaction': [environment_satisfaction],
                'Gender': [gender],
                'HourlyRate': [hourly_rate],
                'JobInvolvement': [job_involvement],
                'JobLevel': [job_level],
                'JobRole': [job_role],
                'JobSatisfaction': [job_satisfaction],
                'MaritalStatus': [marital_status],
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
            input_data['OverTime'] = input_data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)
            
            # Encode categorical data
            x_cat_input = input_data.select_dtypes(include=['object'])
            x_cat_input = encoder.transform(x_cat_input).toarray()
            x_cat_input = pd.DataFrame(x_cat_input)
            
            # Combine the categorical and numerical features
            x_num_input = input_data.select_dtypes(include=['float', 'int'])
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
    st.subheader("Upload a CSV file to predict employee attrition.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            
            # Drop unnecessary columns if they exist
            new_data = new_data.drop(columns=columns_to_drop, axis=1, errors='ignore')
            
            st.write("Uploaded Data (first 5 rows):")
            st.write(new_data.head())
            
            # Pre-process the data
            new_data['OverTime'] = new_data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)
            
            # Encode categorical data
            x_cat_new = new_data.select_dtypes(include=['object'])
            x_cat_new = encoder.transform(x_cat_new).toarray()
            x_cat_new = pd.DataFrame(x_cat_new)
            
            # Combine the categorical and numerical features
            x_num_new = new_data.select_dtypes(include=['float', 'int'])
            x_all_new = pd.concat([x_cat_new, x_num_new], axis=1)
            x_all_new.columns = x_all_new.columns.astype(str)
            
            # Scale the features
            x_new = scaler.transform(x_all_new)
            
            # Make predictions
            y_pred_new = model.predict(x_new)
            y_pred_new = (y_pred_new > 0.5).astype(int)
            
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
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please upload a CSV file to continue.")
