import os, sys
from src.exception.exception import AppException
from src.logger.log import logging
from src.utils.utils import load_object
import pandas as pd

class PredictPipleline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            prerprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')
            
            preprocessor = load_object(prerprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise AppException(e, sys) from e

class CustomData:
    def __init__(self,
                step:float,
                amount:float,
                newbalanceOrig:float,
                newbalanceDest:float,
                isFlaggedFraud:float):
        self.step=step
        self.amount=amount
        self.newbalanceOrig=newbalanceOrig
        self.newbalanceDest=newbalanceDest
        self.isFlaggedFraud=isFlaggedFraud

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'step':[self.step],
                'amount':[self.amount],
                'newbalanceOrig':[self.newbalanceOrig],
                'newbalanceDest':[self.newbalanceDest],
                'isFlaggedFraud':[self.isFlaggedFraud]
                
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise AppException(e, sys) from e    