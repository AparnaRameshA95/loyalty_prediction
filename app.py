from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained models and encoders
scaler = joblib.load('robust_scaler1.pkl')
encoder = joblib.load('onehot_encoder1.pkl')
target_encoder = joblib.load('target_encoder1.pkl')
rfe_model = joblib.load('rfe_model1.pkl')
best_model = joblib.load('best_model1.pkl')

# Feature list for proper input mapping
selected_features = ['Age', 'Items Purchased', 'Total Spent', 'Discount (%)',
       'Satisfaction Score', 'Revenue']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON data
    data = request.json
    try:
        # Convert to DataFrame for processing
        input_data = pd.DataFrame([data])

        # Scale numerical features
        input_data[['Total Spent', 'Discount (%)', 'Revenue']] = scaler.transform(
            input_data[['Total Spent', 'Discount (%)', 'Revenue']]
        )

        # One-hot encode categorical features
        categorical_data = input_data[['Gender', 'Region', 'Product Category', 'Preferred Visit Time', 'Payment Method']]
        encoded_categorical = encoder.transform(categorical_data)
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical, 
            columns=encoder.get_feature_names_out(categorical_data.columns)
        )

        # Combine scaled and encoded features
        input_data = pd.concat([input_data.drop(columns=categorical_data.columns), encoded_categorical_df], axis=1)

        # Select RFE-selected features
        input_data = input_data[selected_features]

        # Make prediction
        prediction = best_model.predict(input_data)
        category = target_encoder.inverse_transform(prediction)[0]

        return jsonify({'Loyalty Category': category})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
