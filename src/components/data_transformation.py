import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import DATA_DIR, DATASET_NAME, PREPROCESSOR_PATH
from src.exception import CustomException
from src.logger import logging
from src.utils.common import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = PREPROCESSOR_PATH
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This Function Creates Pipeline:
        1. For Numerical columns (Imputation + Scaling)
        2. For Categorical columns (Imputation + OneHotEncoding)
        '''
        try:
            df = pd.read_csv(f'{DATA_DIR}/{DATASET_NAME}')
            target_column_name = 'FraudFound_P'
            
            # Drop Target and PolicyNumber Column
            df_features = df.drop(columns=['PolicyNumber', target_column_name])
            
            numerical_columns = df_features.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = df_features.select_dtypes(include=['str', 'object']).columns
            
            # Numerical Pipeline
            # Fix Missing Values by median and then do Standardize
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Categorical Pipeline
            # If Missing Values then fill it with 'most_frequent' and then do OneHotEncode
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # we used sparse=False because we want to avoid creating a sparse matrix 
                    # for the concatenation with the array in the next step train_arr, test_arr
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            # Manual Cleaning
            logging.info("Performing Manual Cleaning based on EDA...")
            
            # Dropping '0' in DayOfWeekClaimed
            train_df = train_df[train_df['DayOfWeekClaimed'] != '0']
            test_df = test_df[test_df['DayOfWeekClaimed'] != '0']
            
            # Dropping PolicyNumber Column
            train_df = train_df.drop(columns=['PolicyNumber'])
            test_df = test_df.drop(columns=['PolicyNumber'])
            
            # Set Target Column
            target_column_name = 'FraudFound_P'
            drop_columns = [target_column_name]
            
            # Seperate X and y
            
            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]
            
            # Applying Preprocessor
            preprocessing_obj = self.get_data_transformer_object()
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            # fit_transform on TRAIN, transform on TEST
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Concatenate with Target (Save array)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            # Save Pickle File
            logging.info(f"Saving preprocessing object...")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)