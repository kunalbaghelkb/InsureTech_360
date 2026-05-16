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
            # --- PERFECT ALIGNMENT: Drop 'Kill Switches' that overshadow risk signals ---
            kill_switches = ['PolicyNumber', 'PoliceReportFiled', 'WitnessPresent', 'AgentType']
            df_features = df.drop(columns=[target_column_name] + kill_switches)
            
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
            
            # Categorical Pipeline (One-Hot)
            # NO SCALING for dummy variables to avoid signal inflation/deflation
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )
            
            # Ordinal Pipeline (Mapped columns)
            # NO SCALING to preserve the integer meaning (0, 1, 2...)
            ord_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler(with_std=False)) # Only center, don't scale variance
                ]
            )
            
            # Recalculate columns after potential manual mapping
            numerical_columns = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df_features.select_dtypes(include=['str', 'object']).columns.tolist()
            
            # Explicitly move mapped columns to numerical list if they were caught as categorical
            ordinal_cols = ['AgeOfVehicle', 'VehiclePrice', 'AgeOfPolicyHolder', 'PastNumberOfClaims', 
                            'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars', 
                            'Days_Policy_Accident', 'Days_Policy_Claim']
            
            for col in ordinal_cols:
                if col in categorical_columns:
                    categorical_columns.remove(col)
                if col not in numerical_columns:
                    numerical_columns.append(col)

            logging.info(f"Final Categorical columns: {categorical_columns}")
            logging.info(f"Final Numerical columns: {numerical_columns}")
            
            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, [c for c in numerical_columns if c not in ordinal_cols]),
                    ("ord_pipeline", ord_pipeline, ordinal_cols),
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
            
            # --- PERFECT ALIGNMENT: Prune Kill Switches ---
            kill_switches = ['PoliceReportFiled', 'WitnessPresent', 'AgentType']
            train_df = train_df.drop(columns=kill_switches)
            test_df = test_df.drop(columns=kill_switches)
            
            drop_columns = [target_column_name]
            
            # --- PERFECT ALIGNMENT: Ordinal Mapping ---
            logging.info("Applying Ordinal Mapping and Make Grouping...")
            
            # Group high-end makes to avoid 0% fraud trap
            luxury_makes = ['BMW', 'Lexus', 'Ferrari', 'Jaguar', 'Porche', 'Mecedes']
            train_df['Make'] = train_df['Make'].apply(lambda x: 'Luxury' if x in luxury_makes else x)
            test_df['Make'] = test_df['Make'].apply(lambda x: 'Luxury' if x in luxury_makes else x)
            
            ordinal_map = {
                'AgeOfVehicle': {'new': 0, '2 years': 1, '3 years': 2, '4 years': 3, '5 years': 4, '6 years': 5, '7 years': 6, 'more than 7': 7},
                'VehiclePrice': {'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2, '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5},
                'AgeOfPolicyHolder': {'16 to 17': 0, '18 to 20': 1, '21 to 25': 2, '26 to 30': 3, '31 to 35': 4, '36 to 40': 5, '41 to 50': 6, '51 to 65': 7, 'over 65': 8},
                'PastNumberOfClaims': {'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3},
                'NumberOfSuppliments': {'none': 0, '1 to 2': 1, '3 to 5': 2, 'more than 5': 3},
                'AddressChange_Claim': {'no change': 0, 'under 6 months': 1, '1 year': 2, '2 to 3 years': 3, '4 to 8 years': 4},
                'NumberOfCars': {'1 vehicle': 0, '2 vehicles': 1, '3 to 4': 2, '5 to 8': 3, 'more than 8': 4},
                'Days_Policy_Accident': {'none': 0, '1 to 7': 1, '8 to 15': 2, '15 to 30': 3, 'more than 30': 4},
                'Days_Policy_Claim': {'none': 0, '1 to 7': 1, '8 to 15': 2, '15 to 30': 3, 'more than 30': 4}
            }
            
            for col, mapping in ordinal_map.items():
                train_df[col] = train_df[col].map(mapping).fillna(0)
                test_df[col] = test_df[col].map(mapping).fillna(0)
            
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
            logging.info("Saving preprocessing object...")
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