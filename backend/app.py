
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
superkart_api = Flask("superkart_sales_api")
CORS(superkart_api)

# Load the trained model pipeline (preprocessing + model)
model = joblib.load("superkart_sales_forecast_model_v1_0.joblib")

# Health check route
@superkart_api.get('/')
def home():
    return "✅ Welcome to the SuperKart Sales Prediction API"

# Prediction route (Online Inference)
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Parse JSON payload
        data = request.get_json()
        print("Raw incoming data:", data)

        # Validate expected fields
        required_fields = [
            'Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Store_Age_Years',
            'Product_Type_Category'
        ]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f"Missing fields: {missing_fields}"}), 400

        # Convert and transform input
        sample = {
            'Product_Weight': float(data['Product_Weight']),
            'Product_Sugar_Content': data['Product_Sugar_Content'],
            'Product_Allocated_Area': np.log1p(float(data['Product_Allocated_Area'])),  
            'Product_MRP': float(data['Product_MRP']),
            'Store_Size': data['Store_Size'],
            'Store_Location_City_Type': data['Store_Location_City_Type'],
            'Store_Type': data['Store_Type'],
            'Store_Age_Years': int(data['Store_Age_Years']),
            'Product_Type_Category': data['Product_Type_Category']
        }

        input_df = pd.DataFrame([sample])
        print("Transformed input for model:\n", input_df)

        # Make prediction
        prediction = model.predict(input_df).tolist()[0]
        return jsonify({'Predicted_Sales': prediction})

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Prediction route (Batch Inference)
@superkart_api.post('/v1/predictbatch')
def predict_batch_sales():
    try:
        # Validate that the file parameter exists in the incoming multipart request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
            
        file = request.files['file']
        
        # Read the uploaded CSV bytes stream directly into a pandas DataFrame
        input_df = pd.read_csv(file)
        
        # Apply the required log1p step on the area column
        if 'Product_Allocated_Area' in input_df.columns:
            input_df['Product_Allocated_Area'] = np.log1p(input_df['Product_Allocated_Area'].astype(float))
            
        # Define the exact feature columns your model pipeline expects
        expected_fields = [
            'Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Store_Age_Years',
            'Product_Type_Category'
        ]
        
        # Check if any required feature columns are missing from the CSV
        missing_cols = [col for col in expected_fields if col not in input_df.columns]
        if missing_cols:
            return jsonify({'error': f"CSV is missing required feature columns: {missing_cols}"}), 400
            
        # Filter the DataFrame to ONLY include expected features
        # (This safely removes 'Product_Id_char' or any extra tracking indices)
        final_df = input_df[expected_fields]
        
        # Generate predictions for all rows
        predictions = model.predict(final_df).tolist()
        
        # Format response mapping index string to prediction
        response_dict = {str(i): pred for i, pred in enumerate(predictions)}
        return jsonify(response_dict)
        
    except Exception as e:
        print("❌ Error during batch prediction:", str(e))
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

# Run the app (for local testing only)
if __name__ == '__main__':
    superkart_api.run(debug=True)
