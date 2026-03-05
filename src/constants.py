import os

# Base Directory
ARTIFACTS_DIR = "artifacts"
DATA_DIR = os.path.join(ARTIFACTS_DIR, "data")
CAR_IMAGES_DIR = os.path.join(DATA_DIR, "car_images")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PREPROCESSOR_DIR = os.path.join(ARTIFACTS_DIR, "preprocessor")

# Data Paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw.csv")

# CNN Data Paths
CAR_IMAGES_TRAIN_PATH = os.path.join(CAR_IMAGES_DIR, "training")
CAR_IMAGES_TEST_PATH = os.path.join(CAR_IMAGES_DIR, "validation")

# Image Configuration
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
BATCH_SIZE = 32

# Dataset
DATASET_NAME = "insurance_claims.csv"

# Model Paths
ANN_MODEL_PATH = os.path.join(MODELS_DIR, "ann_model.keras")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")

# Preprocessor Path
PREPROCESSOR_PATH = os.path.join(PREPROCESSOR_DIR, "preprocessor.pkl")

# RAG / GenAI
FAISS_DB_PATH = os.path.join(ARTIFACTS_DIR, "faiss_index")
POLICY_PDF_PATH = os.path.join(DATA_DIR, "policy.pdf")