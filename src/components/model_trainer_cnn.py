import os
import sys
from dataclasses import dataclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from src.constants import (
    CNN_MODEL_PATH,
    CAR_IMAGES_TRAIN_PATH,
    CAR_IMAGES_TEST_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    BATCH_SIZE,
)
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
        logging.info("Entered the Custom CNN Model Trainer component (Optimized Scratch Model)")
        try:
            IMG_HEIGHT = IMAGE_HEIGHT
            IMG_WIDTH = IMAGE_WIDTH
            BATCH = BATCH_SIZE
            
            # ---> Data Generators (Strong Augmentation for Scratch Model)
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
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
            
            # ---> Build DEEP Custom CNN Architecture
            logging.info("Building Deep Custom CNN Architecture...")
            
            model = Sequential()
            
            # Block 1
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(2, 2))
            
            # Block 2
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(2, 2))
            
            # Block 3
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(2, 2))
            
            # Block 4
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(2, 2))
            
            # Block 5
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(GlobalAveragePooling2D())
            
            # Fully Connected
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            # ---> Training with Schedulers
            logging.info("Starting Custom CNN Training...")
            
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=100, # Scratch models need more epochs to learn from zero
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            # ---> Saving Model
            logging.info(f"Saving Custom CNN Model at {self.config.trained_model_file_path}")
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            model.save(self.config.trained_model_file_path)
            
            logging.info("Custom CNN Training Completed!")
            return history.history['val_accuracy'][-1]
        
        except Exception as e:
            raise CustomException(e, sys)