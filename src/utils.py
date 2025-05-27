import joblib

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