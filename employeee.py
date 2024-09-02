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

# Option to choose input method
input_method = st.radio("Choose input method", ["Single Person", "CSV File"])

if input_method == "Single Person":
    st.subheader("Enter the details for a single person:")

    # Organizing inputs into columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, help="Enter the age of the employee.")
        daily_rate = st.number_input('Daily Rate', min_value=1, help="Enter the daily rate of the employee.")
        distance_from_home = st.number_input('Distance From Home (miles)', min_value=0, help="Enter the distance from home.")
        education = st.selectbox('Education Level', options=[1, 2, 3, 4, 5], help="1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor")

    with col2:
        environment_satisfaction = st.selectbox('Environment Satisfaction', options=[1, 2, 3, 4], help="Rate the environment satisfaction level.")
        hourly_rate = st.number_input('Hourly Rate', min_value=1, help="Enter the hourly rate of the employee.")
        job_involvement = st.selectbox('Job Involvement', options=[1, 2, 3, 4], help="Rate the job involvement level.")
        job_level = st.selectbox('Job Level', options=[1, 2, 3, 4, 5], help="Enter the job level of the employee.")

    with col3:
        job_satisfaction = st.selectbox('Job Satisfaction', options=[1, 2, 3, 4], help="Rate the job satisfaction level.")
        monthly_income = st.number_input('Monthly Income', min_value=1000, help="Enter the monthly income of the employee.")
        num_companies_worked = st.number_input('Number of Companies Worked', min_value=0, help="Enter the number of companies the employee has worked for.")
        over_time = st.radio('Over Time', options=['Yes', 'No'], help="Does the employee work overtime?")

    st.subheader("Additional Details")
    col4, col5 = st.columns(2)
    
    with col4:
        percent_salary_hike = st.number_input('Percent Salary Hike', min_value=0, help="Enter the percentage of salary hike.")
        performance_rating = st.selectbox('Performance Rating', options=[1, 2, 3, 4], help="Rate the performance of the employee.")
        relationship_satisfaction = st.selectbox('Relationship Satisfaction', options=[1, 2, 3, 4], help="Rate the relationship satisfaction.")
        stock_option_level = st.number_input('Stock Option Level', min_value=0, max_value=3, help="Enter the stock option level of the employee.")

    with col5:
        total_working_years = st.number_input('Total Working Years', min_value=0, max_value=40, help="Enter the total number of working years.")
        training_times_last_year = st.number_input('Training Times Last Year', min_value=0, max_value=10, help="Enter the number of training times last year.")
        work_life_balance = st.selectbox('Work Life Balance', options=[1, 2, 3, 4], help="Rate the work-life balance.")
        years_at_company = st.number_input('Years at Company', min_value=0, max_value=40, help="Enter the number of years at the company.")
        years_in_current_role = st.number_input('Years in Current Role', min_value=0, max_value=20, help="Enter the number of years in the current role.")
        years_since_last_promotion = st.number_input('Years Since Last Promotion', min_value=0, max_value=15, help="Enter the number of years since last promotion.")
        years_with_curr_manager = st.number_input('Years With Current Manager', min_value=0, max_value=15, help="Enter the number of years with the current manager.")

    # Validation to ensure correct logical input
    if years_at_company > total_working_years:
        st.error("Years at Company cannot be greater than Total Working Years.")
    elif st.button('Predict'):
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

        x_cat_input = input_data.select_dtypes(include=['object'])
        x_cat_input = encoder.transform(x_cat_input).toarray()
        x_cat_input = pd.DataFrame(x_cat_input)

        x_num_input = input_data.select_dtypes(include=['float', 'int'])
        x_all_input = pd.concat([x_cat_input, x_num_input], axis=1)
        x_all_input.columns = x_all_input.columns.astype(str)

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
        st.write("Uploaded Data:")
        st.write(new_data.head())

        new_data['OverTime'] = new_data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)

        x_cat_new = new_data.select_dtypes(include=['object'])
        x_cat_new = encoder.transform(x_cat_new).toarray()
        x_cat_new = pd.DataFrame(x_cat_new)

        x_num_new = new_data.select_dtypes(include=['float', 'int'])
        x_all_new = pd.concat([x_cat_new, x_num_new], axis=1)
        x_all_new.columns = x_all_new.columns.astype(str)

        x_new = scaler.transform(x_all_new)

        y_pred_new = model.predict(x_new)
        y_pred_new = (y_pred_new > 0.5).astype(int)

        predictions = pd.DataFrame(y_pred_new, columns=['Attrition Prediction'])
        st.write("Predictions:")
        st.write(predictions)

        csv = predictions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
    else:
        st.write("Please upload a CSV file to continue.")
