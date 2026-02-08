import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.utils.common import load_object
from src.logger import logging
from src.exception import CustomException
from src.constants import *

class ModelEvaluation:
    def __init__(self):
        pass

    def initiate_fraud_evaluation(self):
        """
        Evaluates the ANN Model (Fraud Detection) on Test Data.
        """
        try:
            logging.info("Starting Fraud Model Evaluation...")

            # ---> Load Data & Artifacts
            logging.info("Loading Test Data, Preprocessor, and Model...")
            test_df = pd.read_csv(TEST_DATA_PATH)
            preprocessor = load_object(file_path=PREPROCESSOR_PATH)
            model = load_model(ANN_MODEL_PATH)

            # ---> Separate Features & Target
            logging.info("Preprocessing Test Data...")
            
            cols_to_drop = ["PolicyNumber"]
            target_column_name = "FraudFound_P"
            
            logging.info(f"Dropping ID columns: {cols_to_drop}")
            
            if "PolicyNumber" in test_df.columns:
                test_df = test_df.drop(columns=cols_to_drop)

            # --->Transform Data
            logging.info("Splitting Features and Target...")
            # X_test (Input)
            input_feature_test_df = test_df.drop(columns=[target_column_name])

            # y_test (Actual Answer)
            target_feature_test_df = test_df[target_column_name]
            
            # ---> Transform Data
            test_arr = preprocessor.transform(input_feature_test_df)
            
            # ---> Prediction
            logging.info("Running Predictions on Test Data...")
            y_pred_prob = model.predict(test_arr)
            
            # Threshold 0.5 for Class Weights Model
            y_pred = (y_pred_prob > 0.5).astype("int32")

            # ---> Metrics Calculation
            accuracy = accuracy_score(target_feature_test_df, y_pred)
            cm = confusion_matrix(target_feature_test_df, y_pred)
            cr = classification_report(target_feature_test_df, y_pred)

            logging.info(f"Fraud Model Accuracy: {accuracy}")
            logging.info(f"Confusion Matrix:\n{cm}")
            logging.info(f"Classification Report:\n{cr}")
            
            print("\n------Fraud Model Report------")
            print(f"Accuracy: {accuracy}")
            print(cr)
            print("---------------------------------")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_damage_evaluation(self):
        """
        Evaluates the CNN Model (Car Damage) on Validation Images.
        """
        try:
            logging.info("Starting Damage Model Evaluation...")

            # 1. Load Model
            model = load_model(CNN_MODEL_PATH)

            # 2. Prepare Data Generator (Only Rescaling)
            eval_datagen = ImageDataGenerator(rescale=1./255)

            eval_generator = eval_datagen.flow_from_directory(
                CAR_IMAGES_TEST_PATH,
                target_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=False # Shuffle False for predictions to run in order
            )

            # 3. Evaluate
            logging.info("Evaluating CNN Model...")
            score = model.evaluate(eval_generator, verbose=1)
            
            loss, accuracy = score[0], score[1]

            logging.info(f"CNN Model - Loss: {loss}, Accuracy: {accuracy}")
            
            print("\n------- DAMAGE MODEL REPORT ----------")
            print(f"Loss: {loss}")
            print(f"Accuracy: {accuracy}")
            print("----------------------------------------")

        except Exception as e:
            raise CustomException(e, sys)