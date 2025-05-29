import joblib
import numpy as np
import pandas as pd
import scipy


def load_joblib(file_path):
     """
     Load a joblib file from the specified path.
     
     Args:
          file_path (str): The path to the joblib file.
          
     Returns:
          object: The object loaded from the pickle file.
     """
     with open(file_path, 'rb') as file:
          return joblib.load(file)


def preprocessing(input_data, encoder, sc, mimx):
     """
     Preprocess the input data for model prediction.
     
     Args:
          input_data (dict): The input data containing user information.
          encoder (object): The encoder for categorical features.
          sc (object): The scaler for continuous features.
          mimx (object): The mimx transformer for specific features.
          
     Returns:
          dict: The preprocessed input data ready for model prediction.
     """
     # Step 1: Define model columns
     model_columns = ['age', 'workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

     input_dict = {col: 0 for col in model_columns}

     # Step 4: Fill continuous and categorical fields
     input_dict['age'] = int(input_data.get('age', 0))
     input_dict['workclass'] = (input_data.get('workclass', 0))
     input_dict['educational-num'] = int(input_data.get('educational_num', 0))
     input_dict['marital-status'] = (input_data.get('marital-status', 0))
     input_dict['occupation'] = (input_data.get('occupation', 0))
     input_dict['relationship'] = (input_data.get('relationship', 0))
     input_dict['race'] = (input_data.get('race', 0))
     input_dict['gender'] = (input_data.get('gender', 0))
     input_dict['capital-gain'] = int(input_data.get('capital-gain', 0))
     input_dict['capital-loss'] = int(input_data.get('capital-loss', 0))
     input_dict['hours-per-week'] = int(input_data.get('hours-per-week', 0))
     input_dict['native-country'] = (input_data.get('native-country', 0))

     # Step 5: Create DataFrame, scale, predict
     df_input = pd.DataFrame([input_dict])

     num_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
     cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

     # label encoding categorical features
     for col in cat_cols:
          df_input[col] = encoder.fit_transform(df_input[col])

     # scaling numerical features
     skewed_cols = ['age', 'hours-per-week']
     normal_cols = ['capital-gain', 'capital-loss']
     
     df_input[normal_cols] = sc.fit_transform(df_input[normal_cols])
     df_input[skewed_cols] = mimx.fit_transform(df_input[skewed_cols])

     return df_input