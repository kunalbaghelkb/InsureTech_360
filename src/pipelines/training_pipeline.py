import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer_cnn import ModelTrainerCNN
from src.components.model_evaluation import ModelEvaluation
from src.components.vector_db_builder import VectorDBBuilder
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def run_fraud_pipeline(self):
        """
        Runs the complete pipeline for Fraud Detection (ANN)
        """
        try:
            logging.info("Started Fraud Training Pipeline")
            
            # Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # Data Transformation (Scaling/Encoding)
            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

            # Model Training (ANN)
            trainer = ModelTrainer()
            accuracy, _ = trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Fraud Model Trained with Accuracy: {accuracy}")
            
            # Model Evaluation
            evaluator = ModelEvaluation()
            evaluator.initiate_fraud_evaluation()
            logging.info("Fraud Model Evaluation Completed!")

        except Exception as e:
            raise CustomException(e, sys)

    def run_damage_pipeline(self):
        """
        Runs the complete pipeline for Car Damage (CNN)
        """
        try:
            logging.info("Started Damage Training Pipeline")
            
            # Model Training: CNN Direct Takes Image From Folder, Transformation handles internally
            trainer_cnn = ModelTrainerCNN()
            val_acc = trainer_cnn.initiate_model_trainer()
            logging.info(f"CNN Model Trained with Validation Accuracy: {val_acc}")
            
            # Model Evaluation
            evaluator = ModelEvaluation()
            evaluator.initiate_damage_evaluation()
            logging.info("Damage Model Evaluation Completed!")

        except Exception as e:
            raise CustomException(e, sys)
        
    def run_rag_ingestion(self):
        """
        Runs the RAG Pipeline: Ingest PDF and generates Create Vector DB
        """
        try:
            logging.info("Started RAG Vector DB Pipeline")
            
            builder = VectorDBBuilder()
            builder.create_vector_db()
            
            logging.info("RAG Vector DB Pipeline Completed!")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test Run
    pipeline = TrainPipeline()
    pipeline.run_fraud_pipeline()
    pipeline.run_damage_pipeline()
    pipeline.run_rag_ingestion()