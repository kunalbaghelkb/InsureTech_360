import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.exception import CustomException
from src.logger import logging
from src.utils.common import load_object
from src.constants import (
    ANN_MODEL_PATH,
    PREPROCESSOR_PATH,
    CNN_MODEL_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

# Fraud Prediction Pipeline
class FraudData:
    """
    Maps HTML form fields to a Pandas DataFrame using the EXACT column names
    from the training CSV. No hardcoded defaults — every value comes from the form.
    """
    def __init__(self,
        # ── Numeric ─────────────────────────────────────────────────────────
        age: int,
        week_of_month: int,
        week_of_month_claimed: int,
        rep_number: int,
        deductible: int,
        driver_rating: int,
        year: int,
        # ── Categorical — main fraud signals ────────────────────────────────
        month: str,
        day_of_week: str,
        make: str,
        accident_area: str,
        day_of_week_claimed: str,
        month_claimed: str,
        sex: str,
        marital_status: str,
        fault: str,
        policy_type: str,
        vehicle_category: str,
        vehicle_price: str,
        days_policy_accident: str,
        days_policy_claim: str,
        past_number_of_claims: str,
        age_of_vehicle: str,
        age_of_policy_holder: str,
        police_report_filed: str,
        witness_present: str,
        agent_type: str,
        number_of_suppliments: str,
        address_change_claim: str,
        number_of_cars: str,
        base_policy: str,
    ):
        # Keys MUST match the exact CSV column names (what training used)
        self.data_dict = {
            "Month":                  [month],
            "WeekOfMonth":            [week_of_month],
            "DayOfWeek":              [day_of_week],
            "Make":                   [make],
            "AccidentArea":           [accident_area],
            "DayOfWeekClaimed":       [day_of_week_claimed],
            "MonthClaimed":           [month_claimed],
            "WeekOfMonthClaimed":     [week_of_month_claimed],
            "Sex":                    [sex],
            "MaritalStatus":          [marital_status],
            "Age":                    [age],
            "Fault":                  [fault],
            "PolicyType":             [policy_type],
            "VehicleCategory":        [vehicle_category],
            "VehiclePrice":           [vehicle_price],
            "RepNumber":              [rep_number],
            "Deductible":             [deductible],
            "DriverRating":           [driver_rating],
            "Days_Policy_Accident":   [days_policy_accident],
            "Days_Policy_Claim":      [days_policy_claim],
            "PastNumberOfClaims":     [past_number_of_claims],
            "AgeOfVehicle":           [age_of_vehicle],
            "AgeOfPolicyHolder":      [age_of_policy_holder],
            "PoliceReportFiled":      [police_report_filed],
            "WitnessPresent":         [witness_present],
            "AgentType":              [agent_type],
            "NumberOfSuppliments":    [number_of_suppliments],
            "AddressChange_Claim":    [address_change_claim],
            "NumberOfCars":           [number_of_cars],
            "Year":                   [year],
            "BasePolicy":             [base_policy],
        }

    def get_data_as_df(self):
        try:
            df = pd.DataFrame(self.data_dict)
            logging.info(f"Pipeline Input (Mapped): {df.to_dict(orient='records')[0]}")
            
            # --- PERFECT ALIGNMENT: Apply same Mapping and Grouping as Trainer ---
            luxury_makes = ['BMW', 'Lexus', 'Ferrari', 'Jaguar', 'Porche', 'Mecedes']
            if 'Make' in df.columns:
                df['Make'] = df['Make'].apply(lambda x: 'Luxury' if x in luxury_makes else x)

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
                if col in df.columns:
                    df[col] = df[col].map(mapping).fillna(0)
            
            # --- PERFECT ALIGNMENT: Prune Kill Switches ---
            kill_switches = ['PoliceReportFiled', 'WitnessPresent', 'AgentType']
            df = df.drop(columns=[col for col in kill_switches if col in df.columns])
            
            return df
        except Exception as e:
            raise CustomException(e, sys)


class FraudPredictor:
    """
    Loads the ANN model and preprocessor once at class level to avoid
    reloading heavy files on every API request.
    """
    _model = None
    _preprocessor = None

    def __init__(self):
        self.model_path = ANN_MODEL_PATH
        self.preprocessor_path = PREPROCESSOR_PATH

    @classmethod
    def reset(cls):
        """Force reload of model/preprocessor (call after retraining)."""
        cls._model = None
        cls._preprocessor = None

    def _load_resources(self):
        if FraudPredictor._model is None:
            logging.info(f"Loading Fraud ANN model from {self.model_path}")
            FraudPredictor._model = load_model(self.model_path)
        if FraudPredictor._preprocessor is None:
            logging.info(f"Loading preprocessor from {self.preprocessor_path}")
            FraudPredictor._preprocessor = load_object(file_path=self.preprocessor_path)

    def predict(self, features_df):
        try:
            self._load_resources()

            # Reorder columns to match the exact order the preprocessor was fitted on
            expected_cols = list(FraudPredictor._preprocessor.feature_names_in_)
            features_df = features_df[expected_cols]

            # Transform and predict
            data_scaled = FraudPredictor._preprocessor.transform(features_df)
            prediction_prob = FraudPredictor._model.predict(data_scaled)[0][0]
            
            # Return binary class AND raw probability
            # HIGH SENSITIVITY: Lower threshold to 0.35 to catch red flags
            prediction_class = 1 if prediction_prob > 0.35 else 0
            
            # Placeholder for explainability (to be expanded)
            top_features = []
            is_ood = False
            ood_reasons = []

            return prediction_class, float(prediction_prob), top_features, is_ood, ood_reasons

        except Exception as e:
            raise CustomException(e, sys)


# Damage Prediction Pipeline
class DamagePredictor:
    """
    Loads the CNN model once at class level to avoid reloading on every request.
    """
    _model = None

    def __init__(self):
        self.model_path = CNN_MODEL_PATH

    def _load_resources(self):
        if DamagePredictor._model is None:
            logging.info(f"Loading CNN model from {self.model_path}")
            DamagePredictor._model = load_model(self.model_path)

    def predict(self, image_path):
        try:
            self._load_resources()

            # Preprocess Image (Resize & Normalize)
            img = image.load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0                   # Normalize to [0, 1]

            # Predict — class 0 = Damaged, class 1 = Whole
            result = DamagePredictor._model.predict(img_array)
            return "no-damage-detected" if result[0][0] > 0.5 else "damage-detected"

        except Exception as e:
            raise CustomException(e, sys)