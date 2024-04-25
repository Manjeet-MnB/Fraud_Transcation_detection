import os,sys
import pandas as pd
import numpy as np
from src.logger.log import logging
from src.exception.exception import AppException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

# Creating a Data Ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            # reading the data
            df = pd.read_csv(os.path.join('notebooks/data','Fraud.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('train test split')
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info("Error occuerd in Data Ingestion Config")
            raise AppException(e, sys) from e
                