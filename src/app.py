from flask import Flask, logging, request, jsonify, render_template
import logging
from flask_cors import CORS
import pandas as pd
from utils import load_joblib
import os
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
CORS(app)

# Load model and preprocessors
model = load_joblib('/workspaces/ml_techniques/src/models/ds1/best_model_random_forest.pkl')
sc= load_joblib('/workspaces/ml_techniques/src/models/ds1/label_encoder.pkl')
mimx = load_joblib('/workspaces/ml_techniques/src/models/ds1/minmax_scaler.pkl')
print("Loaded model type:", type(model))

columns = ['age', 'educational-num', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay', 'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse', 'marital-status_Married-spouse-absent', 'marital-status_Never-married', 'marital-status_Separated', 'marital-status_Widowed', 'occupation_Adm-clerical', 'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving', 'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']  

class_names = {
     0: "<= 50K",
     1: "> 50K",
}
@app.route("/")
def index():
     return render_template('index.html')

@app.route("/home")
def home():
     return "ML Proj  API is running!"

@app.route("/predict", methods=['POST'])
def predict():
     logging.debug("Received request data: %s", request.data)

     if not request.json or 'data' not in request.json:
          logging.debug("error Invalid input format: %s",400)
     if not isinstance(request.json['data'], list):
          logging.debug("'error: Input data must be a list': %s",400)
     try:
          # Expecting a list of feature values
          input_data = request.get_json("data")
          print(input_data)
          model_columns = [
               'age', 'educational-num', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
               'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
               'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay',
               'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
               'marital-status_Married-spouse-absent', 'marital-status_Never-married', 'marital-status_Separated',
               'marital-status_Widowed',
               'occupation_Adm-clerical', 'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial',
               'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
               'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv',
               'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving',
               'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried',
               'relationship_Wife',
               'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White',
               'native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia',
               'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador',
               'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece',
               'native-country_Guatemala', 'native-country_Haiti', 'native-country_Honduras', 'native-country_Hong',
               'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland',
               'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos',
               'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)',
               'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal',
               'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan',
               'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States',
               'native-country_Vietnam', 'native-country_Yugoslavia'
          ]

          # Step 3: Initialize input dictionary
          input_dict = {col: 0 for col in model_columns}

          # Step 4: Fill continuous and categorical fields
          input_dict['age'] = int(input_data.get('age', 0))
          input_dict['educational-num'] = int(input_data.get('educational_num', 0))
          input_dict['capital-gain'] = int(input_data.get('capital-gain', 0))
          input_dict['capital-loss'] = int(input_data.get('capital-loss', 0))
          input_dict['hours-per-week'] = int(input_data.get('hours-per-week', 0))

          # One-hot encoded categorical fields (only set selected value to 1)
          field_map = {
          'workclass': 'workclass_',
          'marital_status': 'marital-status_',
          'relationship_type': 'relationship_',
          'occupation_type': 'occupation_',
          'race': 'race_',
          'native_country': 'native-country_'
          }

          for field, prefix in field_map.items():
               val = input_data.get(field)
               key = f"{prefix}{val}"
               if key in input_dict:
                    input_dict[key] = 1

          # Debug: Print final input vector
          print("Final input vector:", input_dict)

          # Step 5: Create DataFrame, scale, predict
          df_input = pd.DataFrame([input_dict])

          # pre-processing
          df_input[['age']] = sc.transform(df_input[['age']])
          df_input['gender'] = 1 if input_data.get('gender', '').lower()== 'male' else 0
          df_input[['hours-per-week']] = sc.transform(df_input[['hours-per-week']])
          df_input[['capital-gain']] = mimx.transform(df_input[['capital-gain']])
          df_input[['capital-loss']] = mimx.transform(df_input[['capital-loss']])

          # Ensure the DataFrame has the same columns as the model expects
          df_input= df_input.reindex(columns=model_columns, fill_value=0)
          print("Final input data:", df_input)

          prediction = model.predict(df_input)

          activity_mapping = {
               0: '<=50K',
               1: '>50K',  
          }

          # Step 6: Format and return prediction
          result = int(prediction[0])
          label = activity_mapping.get(result, "Unknown")

          return jsonify({'prediction': result, 'activity': label})

     except Exception as e:
          return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=5000, debug=True)


