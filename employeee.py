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

# Columns in the CSV file
csv_columns = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber',
    'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# Columns to be dropped
columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']

# Filter columns to use in the interface
use_columns = [col for col in csv_columns if col not in columns_to_drop]

# Option to choose input method
input_method = st.radio("Choose input method", ["Single Person", "CSV File"])

# Set options based on the CSV columns
business_travel_options = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
department_options = ['Sales', 'Research & Development', 'Human Resources']
education_field_options = ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']
gender_options = ['Male', 'Female']
job_role_options = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 
                   'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']
marital_status_options = ['Single', 'Married', 'Divorced']
overtime_options = ['Yes', 'No']

if input_method == "Single Person":
    st.subheader("Enter the details for a single person:")

    # Creating input fields dynamically based on the columns that are used
    input_data = {}
    
    for col in use_columns:
        if col in ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobLevel', 'MonthlyIncome', 'MonthlyRate', 
                   'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 
                   'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                   'YearsWithCurrManager']:
            input_data[col] = st.number_input(col, min_value=0)
        elif col == 'BusinessTravel':
            input_data[col] = st.selectbox(col, options=business_travel_options)
        elif col == 'Department':
            input_data[col] = st.selectbox(col, options=department_options)
        elif col == 'Education':
            input_data[col] = st.selectbox(col, options=[1, 2, 3, 4, 5])
        elif col == 'EducationField':
            input_data[col] = st.selectbox(col, options=education_field_options)
        elif col == 'EnvironmentSatisfaction':
            input_data[col] = st.selectbox(col, options=[1, 2, 3, 4])
        elif col == 'Gender':
            input_data[col] = st.selectbox(col, options=gender_options)
        elif col == 'JobInvolvement':
            input_data[col] = st.selectbox(col, options=[1, 2, 3, 4])
        elif col == 'JobRole':
            input_data[col] = st.selectbox(col, options=job_role_options)
        elif col == 'JobSatisfaction':
            input_data[col] = st.selectbox(col, options=[1, 2, 3, 4])
        elif col == 'MaritalStatus':
            input_data[col] = st.selectbox(col, options=marital_status_options)
        elif col == 'OverTime':
            input_data[col] = st.selectbox(col, options=overtime_options)
        elif col == 'PerformanceRating':
            input_data[col] = st.selectbox(col, options=[1, 2, 3, 4])
        elif col == 'RelationshipSatisfaction':
            input_data[col] = st.selectbox(col, options=[1, 2, 3, 4])
        elif col == 'WorkLifeBalance':
            input_data[col] = st.selectbox(col, options=[1, 2, 3, 4])

    # Preprocessing and prediction logic
    if st.button('Predict'):
        # Prepare the input data
        input_df = pd.DataFrame([input_data])

        # Pre-process the data
        input_df['OverTime'] = input_df['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)

        # Encode categorical data
        x_cat_input = input_df.select_dtypes(include=['object'])
        x_cat_input = encoder.transform(x_cat_input).toarray()
        x_cat_input = pd.DataFrame(x_cat_input)

        # Combine the categorical and numerical features
        x_num_input = input_df.select_dtypes(include=['float', 'int'])
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
    else:
        st.write("Please upload a CSV file to continue.")
