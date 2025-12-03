from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import os
from datetime import datetime

app = Flask(__name__)

# Load the model if it exists
model = None
correlation_coefficient = 0.78  # Default correlation between ALB and A/G Ratio

if os.path.exists('model.pkl'):
    try:
        model = joblib.load('model.pkl')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load correlation coefficient if available
if os.path.exists('correlation.txt'):
    try:
        with open('correlation.txt', 'r') as f:
            correlation_coefficient = float(f.read().strip())
        print(f"Correlation coefficient loaded: {correlation_coefficient}")
    except Exception as e:
        print(f"Error loading correlation coefficient: {e}")


@app.route('/')
def home():
    current_year = datetime.now().year
    return render_template('index.html', prediction=None, current_year=current_year)


@app.route('/predict', methods=['POST'])
def predict():
    current_year = datetime.now().year

    if model is None:
        return render_template('index.html',
                               prediction="Model not available. Please train the model first.",
                               prediction_class="text-yellow-600 font-bold",
                               current_year=current_year)

    try:
        # Get form data
        age = int(request.form['age'])
        gender = request.form['gender']
        tb = float(request.form['tb'])
        db = float(request.form['db'])
        alkphos = int(request.form['alkphos'])
        sgpt = int(request.form['sgpt'])
        sgot = int(request.form['sgot'])
        tp = float(request.form['tp'])
        alb = float(request.form['alb'])
        agratio = request.form['agratio']

        # Handle empty A/G Ratio
        if not agratio or agratio.strip() == '':
            agratio = alb * correlation_coefficient
        else:
            agratio = float(agratio)

        # Create DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'TB': [tb],
            'DB': [db],
            'Alkphos': [alkphos],
            'Sgpt': [sgpt],
            'Sgot': [sgot],
            'TP': [tp],
            'ALB': [alb],
            'A/G Ratio': [agratio]
        })

        # One-hot encode Gender
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoder.fit(pd.DataFrame({'Gender': ['Male', 'Female']}))
        gender_encoded = encoder.transform(input_data[['Gender']])
        gender_df = pd.DataFrame(gender_encoded, columns=['Gender_Male'])

        # Combine with other features
        input_processed = pd.concat([
            input_data.drop('Gender', axis=1).reset_index(drop=True),
            gender_df.reset_index(drop=True)
        ], axis=1)

        # Ensure columns are in the correct order
        expected_columns = ['Age', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio', 'Gender_Male']
        input_processed = input_processed[expected_columns]

        # Make prediction
        prediction = model.predict(input_processed)[0]

        # Convert prediction to readable format
        result = "Liver Cirrhosis Detected" if prediction == 1 else "No Liver Cirrhosis Detected"
        result_class = "text-red-600 font-bold" if prediction == 1 else "text-green-600 font-bold"

        return render_template('index.html', prediction=result, prediction_class=result_class,
                               current_year=current_year)

    except Exception as e:
        error_message = f"Error processing prediction: {str(e)}"
        return render_template('index.html', prediction=error_message, prediction_class="text-yellow-600 font-bold",
                               current_year=current_year)


if __name__ == '__main__':
    app.run(debug=True)