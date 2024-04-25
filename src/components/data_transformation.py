import os, sys
from src.logger.log import logging
from src.exception.exception import AppException
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Started")

            # Numerical Features
            numerical_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
            categorical_columns = ['type', 'nameOrig', 'nameDest']

            numerical_columns.remove('oldbalanceOrg')
            numerical_columns.remove('oldbalanceDest')

            categorical_columns.remove('type')
            categorical_columns.remove('nameOrig')
            categorical_columns.remove('nameDest')

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', RobustScaler())
            ])

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns)
            ])
            return preprocessor

        except Exception as e:
            raise AppException(e, sys) from e

    def remote_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            iqr = Q3 - Q1

            upper_limit = Q3 + 1.5 * iqr
            lowwer_limit = Q1 - 1.5 * iqr

            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lowwer_limit), col] = lowwer_limit

            return df
        
        except Exception as e:
            logging.info("Outliers handling code")
            raise AppException(e, sys) from e 

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            numerical_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
            categorical_columns = ['type', 'nameOrig', 'nameDest']

            numerical_columns.remove('oldbalanceOrg')
            numerical_columns.remove('oldbalanceDest')

            categorical_columns.remove('type')
            categorical_columns.remove('nameOrig')
            categorical_columns.remove('nameDest')

            for col in numerical_columns:
                self.remote_outliers_IQR(col = col, df = train_data)

                logging.info("outliers capped on our train data")

            for col in numerical_columns:
                self.remote_outliers_IQR(col = col, df = test_data) 

                logging.info("outliers capped on our test data")

                preprocessor_obj = self.get_data_transformation_obj()

                target_column = 'isFraud'
                drop_columns = [target_column,'oldbalanceOrg','oldbalanceDest','type','nameOrig','nameDest']

                logging.info("splitting train data into dependent and Independent Features")

                input_feature_train_data = train_data.drop(drop_columns, axis=1)
                target_feature_train_data = train_data[target_column]

                logging.info("splitting test data into dependent and Independent Features")

                input_feature_test_data = test_data.drop(drop_columns, axis=1)
                target_feature_test_data = test_data[target_column]

                # Applying transformation on train and test data
                input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
                input_test_arr = preprocessor_obj.transform(input_feature_test_data)

                # Preprocessor obj on our train and test data

                # Apply preprocessor object on our train data and test data
                train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
                test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

                save_object(file_path=self.data_transformation.preprocessor_obj_file_path, obj=preprocessor_obj)
                
                return(train_array, test_array, self.data_transformation.preprocessor_obj_file_path)

        except Exception as e:
            raise AppException(e, sys) from e 
