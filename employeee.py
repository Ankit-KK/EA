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
        age = st.number_input('Age', min_value=18, max_value=100)
        daily_rate = st.number_input('Daily Rate', min_value=1)
        distance_from_home = st.number_input('Distance From Home', min_value=0)
        education = st.selectbox('Education Level', options=[1, 2, 3, 4, 5])

    with col2:
        environment_satisfaction = st.selectbox('Environment Satisfaction', options=[1, 2, 3, 4])
        job_involvement = st.selectbox('Job Involvement', options=[1, 2, 3, 4])
        job_level = st.selectbox('Job Level', options=[1, 2, 3, 4, 5])

    with col3:
        job_satisfaction = st.selectbox('Job Satisfaction', options=[1, 2, 3, 4])
        monthly_income = st.number_input('Monthly Income', min_value=1000)
        over_time = st.radio('Over Time', options=['Yes', 'No'])

    # Validation to ensure correct logical input
    if st.button('Predict'):
        # Prepare the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'DailyRate': [daily_rate],
            'DistanceFromHome': [distance_from_home],
            'Education': [education],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobSatisfaction': [job_satisfaction],
            'MonthlyIncome': [monthly_income],
            'OverTime': [over_time]
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
        new_data = pd.read_csv(uploaded_file)
        
        # Drop unnecessary columns if they exist
        columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
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
