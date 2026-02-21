import sys
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.exception import CustomException
from src.logger import logging
from src.utils.common import load_object
from src.constants import *

# Fraud Prediction Pipeline
class FraudData:
    """
    This class maps HTML form data to a Pandas DataFrame
    """
    def __init__(self, 
                 months_as_customer: int,
                 age: int,
                 policy_deductable: int,
                 policy_annual_premium: float,
                 umbrella_limit: int,
                 capital_gains: int,
                 capital_loss: int,
                 incident_hour_of_the_day: int,
                 number_of_vehicles_involved: int,
                 bodily_injuries: int,
                 witnesses: int,
                 total_claim_amount: int,
                 sex: str, 
                 marital_status: str, 
                 fault: str, 
                 accident_area: str, 
                 police_report_filed: str, 
                 witness_present: str, 
                 vehicle_category: str
                 ):
        
        self.data_dict = {
            "MonthsAsCustomer": [months_as_customer],
            "Age": [age],
            "Deductible": [policy_deductable],
            "PolicyAnnualPremium": [policy_annual_premium],
            "UmbrellaLimit": [umbrella_limit],
            "CapitalGains": [capital_gains],
            "CapitalLoss": [capital_loss],
            "IncidentHourOfTheDay": [incident_hour_of_the_day],
            "NumberOfVehiclesInvolved": [number_of_vehicles_involved],
            "BodilyInjuries": [bodily_injuries],
            "Witnesses": [witnesses],
            "TotalClaimAmount": [total_claim_amount],
            "Sex": [sex],                      
            "MaritalStatus": [marital_status], 
            "Fault": [fault],                   
            "AccidentArea": [accident_area],    
            "PoliceReportFiled": [police_report_filed],
            "WitnessPresent": [witness_present], 
            "VehicleCategory": [vehicle_category],
            "PolicyType": ["Sedan - Collision"], 
            "VehiclePrice": ["20000 to 29000"],
            "RepNumber": [1],
            "Days_Policy_Accident": ["more than 30"],
            "Days_Policy_Claim": ["more than 30"],
            "PastNumberOfClaims": ["none"],
            "AgeOfVehicle": ["7 years"],
            "AgeOfPolicyHolder": ["31 to 35"],
            "AgentType": ["External"],
            "NumberOfSuppliments": ["none"],
            "AddressChange_Claim": ["no change"],
            "NumberOfCars": ["1 vehicle"],
            "Year": [1994],
            "BasePolicy": ["Collision"],
            "Make": ["Honda"],
            "Month": ["Jan"],
            "WeekOfMonth": [3],
            "DayOfWeek": ["Monday"],
            "DayOfWeekClaimed": ["Monday"],
            "MonthClaimed": ["Jan"],
            "WeekOfMonthClaimed": [3],
            "DriverRating": [1]
        }
    def get_data_as_df(self):
        try:
            return pd.DataFrame(self.data_dict)
        except Exception as e:
            raise CustomException(e, sys)

class FraudPredictor:
    def __init__(self):
        self.model_path = ANN_MODEL_PATH
        self.preprocessor_path = PREPROCESSOR_PATH
        
    def predict(self, features_df):
        try:
            # Load Model & Scaler
            model = load_model(self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            # Scale the data
            data_scaled = preprocessor.transform(features_df)
            
            # Predict
            prediction = model.predict(data_scaled)
            
            # 4. Return Result (0 or 1)
            # We useed Class Weights so threshold will be 0.5
            return 1 if prediction[0][0] > 0.5 else 0
        
        except Exception as e:
            raise CustomException(e, sys)
        
# Damage Prediciton Pipeline
class DamagePredictor:
    def __init__(self):
        self.model_path = CNN_MODEL_PATH
        
    def predict(self, image_path):
        try:
            logging.info(f"Loading CNN Model from {self.model_path}")
            model = load_model(self.model_path)
            
            # Preprocess Image (Resize & Rescale)
            img = image.load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Batch dimension
            img_array = img_array / 255.0 # Normalize (Standard for CNNs)
            
            # Predict
            result = model.predict(img_array)
            
            # Logic check (0=Damaged or 1=Whole based on training mapping)
            # Usually: 0 (Damage) class comes first, then 1 (Whole)
            if result[0][0] > 0.5:
                return "no-damage-detected"
            else:
                return "damage-detected"
        except Exception as e:
            raise CustomException(e, sys)