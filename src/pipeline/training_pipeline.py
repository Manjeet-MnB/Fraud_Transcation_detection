import os, sys
import pandas as pd
from src.logger.log import logging
from src.exception.exception import AppException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

#src\pipeline\training_pipeline.py

'''class TrainingPipeline:
    def __init__(self):
        try:
            self.data_ingestion = DataIngestion()
            self.data_transformation = DataTransformation()
            self.model_trainer = ModelTrainer()
        except Exception as e:
            raise AppException(e, sys) from e
        
    def start_training_pipeline(self):
        try:
            self.data_ingestion.initiate_data_ingestion()
            self.data_transformation.initiate_data_transformation()
            self.model_trainer.inititate_model_trainer()
        except Exception as e:
            raise AppException(e, sys) from e'''    

if __name__=='__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    model_training = ModelTrainer()
    model_training.inititate_model_trainer(train_array, test_array)