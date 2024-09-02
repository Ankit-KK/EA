import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# Load the pre-trained encoder and scaler
encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming you've saved your scaler similarly

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Set the title of the app
st.title("Employee Attrition Prediction")

# Instruction text
st.write("Upload a CSV file to predict employee attrition.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(new_data.head())  # Display the first few rows of the uploaded data

    # Pre-process the data
    new_data = new_data.drop('Over18', axis=1)
    new_data['OverTime'] = new_data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)
    new_data = new_data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1)

    # Separate categorical and numerical data
    x_cat_new = new_data.select_dtypes(include=['object'])

    # Apply OneHotEncoder (use the same encoder that was fitted on the training data)
    x_cat_new = encoder.transform(x_cat_new).toarray()
    x_cat_new = pd.DataFrame(x_cat_new)

    # Select numerical columns
    x_num_new = new_data.select_dtypes(include=['float', 'int'])

    # Combine the categorical and numerical features
    x_all_new = pd.concat([x_cat_new, x_num_new], axis=1)

    # Ensure all column names are strings
    x_all_new.columns = x_all_new.columns.astype(str)

    # Scale the features (use the same scaler that was fitted on the training data)
    x_new = scaler.transform(x_all_new)

    # Make predictions on the new data
    y_pred_new = model.predict(x_new)

    # Convert predictions to binary values if it's a binary classification problem
    y_pred_new = (y_pred_new > 0.5).astype(int)  # Convert to 0 and 1

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
