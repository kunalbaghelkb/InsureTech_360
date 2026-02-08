import os
import sys
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from src.constants import *
from src.exception import CustomException
from src.logger import logging
from warnings import filterwarnings
filterwarnings('ignore')

@dataclass
class ModelTrainerCNNConfig:
    trained_model_file_path = CNN_MODEL_PATH
    train_data_path = CAR_IMAGES_TRAIN_PATH
    test_data_path = CAR_IMAGES_TEST_PATH
    
class ModelTrainerCNN:
    def __init__(self):
        self.config = ModelTrainerCNNConfig()
        
    def initiate_model_trainer(self):
        logging.info("Entered the CNN Model Trainer component")
        try:
            # Image Parameters
            IMG_HEIGHT = IMAGE_HEIGHT
            IMG_WIDTH = IMAGE_WIDTH
            BATCH = BATCH_SIZE
            
            # ---> Data Generators
            logging.info("Creating Data Generators (Augmentation)...")
            
            train_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True
            )
            
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            train_generator = train_datagen.flow_from_directory(
                self.config.train_data_path,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH,
                class_mode='binary'
            )
            
            val_generator = val_datagen.flow_from_directory(
                self.config.test_data_path,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH,
                class_mode='binary'
            )
            
            # ---> Build ANN Architecture
            logging.info("Building CNN Architecture...")
            
            model = Sequential()
            
            # Conv Layer 1
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
            model.add(MaxPooling2D(2, 2))
            
            # Conv Layer 2
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(2, 2))
            
            # Conv Layer 3
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(2, 2))
            
            # Flatten & Dense
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # ---> Training
            logging.info("Starting CNN Training...")
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=15, 
                callbacks=[early_stop],
                verbose=1
            )
            
            # ---> Saving Model
            logging.info(f"Saving CNN Model at {self.config.trained_model_file_path}")
            
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            model.save(self.config.trained_model_file_path)
            
            logging.info("CNN Model Training Completed!")
            
            # Return Validation Accuracy of last epoch
            return history.history['val_accuracy'][-1]
        
        except Exception as e:
            raise CustomException(e, sys)