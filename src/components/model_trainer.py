import os
import sys
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
            
            # ---> Optimized Class Weights (1:20 Ratio for MAXIMUM Sensitivity)
            logging.info("Applying 1:20 Class Weighting to shatter 'Safe Haven' biases...")
            
            # We use an aggressive ratio to force the model to prioritize risk
            weights_dict = {0: 1.0, 1: 20.0}
            
            logging.info(f"Custom Class Weights: {weights_dict}")
            
            # Note: We are no longer using SMOTE here to keep the data 'Real' but forcing
            # the ANN to treat each fraud case as 10x more important.
            X_train_res, y_train_res = X_train, y_train
            
            # ---> Build High-Precision Deep ANN Architecture
            logging.info("Building Deep ANN Architecture (Optimized for Balanced Prediction)...")
            from tensorflow.keras.layers import BatchNormalization, LeakyReLU
            
            model = Sequential()
            
            # Input Layer
            model.add(Dense(units=128, input_dim=X_train.shape[1]))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(Dropout(0.4))
            
            # Hidden Layer 1
            model.add(Dense(units=64))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(Dropout(0.3))
            
            # Hidden Layer 2
            model.add(Dense(units=32))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(Dropout(0.2))
            
            # Output Layer
            model.add(Dense(units=1, activation='sigmoid'))

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # ---> Train Model
            logging.info("Training ANN Model with SMOTETomek & Adaptive Learning Rate...")
            from tensorflow.keras.callbacks import ReduceLROnPlateau
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
            
            model.fit(
                X_train_res, y_train_res,
                validation_split=0.2,
                epochs=100,
                batch_size=64,
                class_weight=weights_dict,
                callbacks=[early_stop, reduce_lr],
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