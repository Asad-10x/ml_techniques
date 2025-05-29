from flask import Flask, logging, request, jsonify, render_template
import logging
from flask_cors import CORS
import pandas as pd
import utils
import os
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
CORS(app)

# Load model and preprocessors
model = utils.load_joblib('src/models/ds1/best_model_random_forest.pkl')
sc= utils.load_joblib('src/models/ds1/standard_scaler.pkl')
mimx= utils.load_joblib('src/models/ds1/minmax_scaler.pkl')
le = utils.load_joblib('src/models/ds1/label_encoder.pkl')
print("Loaded model type:", type(model))

columns = ['age', 'workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']  

class_names = {
    0: "<= 50K",
    1: "> 50K",
}
@app.route("/")
def index():
        return render_template('index.html')

@app.route('/home')
def home():
        return "ML Proj  API is running!"

@app.route('/predict', methods=['POST'])
def predict():
        if not request.json or 'data' not in request.json:
            return jsonify({'error': 'Invalid input format'}), 400
        if not isinstance(request.json['data'], list):
            return jsonify({'error': 'Input data must be a list'}), 400
        try:
            # Expecting a list of feature values
            input_data = request.get_json('data')
            print(input_data)

            final_input = utils.preprocessing(input_data, le, sc, mimx)

            # Debug: Print final input vector
            print("Final input vector:", final_input)

            # Ensure the DataFrame has the same columns as the model expects
            # final_input = final_input.reindex(columns=model_columns, fill_value=0)
            print("Final input data:", final_input)

            prediction = model.predict(final_input)

            activity_mapping = {
                0: '<=50K',
                1: '>50K',  
            }

            # Step 6: Format and return prediction
            result = int(prediction[0])
            label = activity_mapping.get(result, "Unknown")
            print("Prediction result:", result)

            return jsonify({'prediction': result, 'activity': label})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
        app.run(host="0.0.0.0", port=5000, debug=True)


