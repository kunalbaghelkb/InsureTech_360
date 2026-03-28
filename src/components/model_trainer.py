import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.constants import ANN_MODEL_PATH
from src.exception import CustomException
from src.logger import logging
from warnings import filterwarnings
filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = ANN_MODEL_PATH
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # Array Slicing
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], # X_train = 80%
                train_array[:, -1], # y_train = 80%
                test_array[:, :-1], # X_test = 20%
                test_array[:, -1] #  y_test = 20%
            )
            
            # ---> Calculate Class Weights
            logging.info("Calculating Class Weights to handle imbalance...")
            
            class_weights_vals = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            
            # Keras need dictionary format
            weights_dict = {0: class_weights_vals[0], 1: class_weights_vals[1]}
            
            logging.info(f"Class Weights Calculated: {weights_dict}")
            
            # ---> Build ANN Architecture
            logging.info("Building ANN Architecture...")
            
            model = Sequential()
            
            model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
            model.add(Dropout(0.3))
            
            model.add(Dense(units=32, activation='relu'))
            model.add(Dropout(0.3))
            
            model.add(Dense(units=1, activation='sigmoid'))

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # ---> Train Model
            logging.info("Training ANN Model with Class Weights...")
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                class_weight=weights_dict,
                callbacks=[early_stop],
                verbose=1
            )
            
            # ---> Evaluation
            logging.info("Evaluating Model on Test Data...")
            
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype("int32")
            
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            
            logging.info(f"Test Accuracy: {acc}")
            logging.info(f"Confusion Matrix:\n{cm}")
            logging.info(f"Classification Report:\n{cr}")
            
            # ---> Save Model
            logging.info("Saving Trained Model...")
            
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            model.save(self.model_trainer_config.trained_model_file_path)
            
            return (
                acc,
                self.model_trainer_config.trained_model_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)