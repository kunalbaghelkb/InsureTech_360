import os
import sys
import pickle
from src.exception import CustomException

def save_object(file_path, obj):
    '''
    This Function Saves Python Object (Model/Preprocessor) in Pickel Format
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    '''
    This Function Load Pickled File Object
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)