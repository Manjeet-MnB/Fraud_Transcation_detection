import os, sys
import pandas as pd
import numpy as np
from src.logger.log import logging
from src.exception.exception import AppException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.utils import save_object
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.utils.utils import save_object
from src.utils.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_apth = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting our data into dependent and independent features")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model = {
                "Random Forest": RandomForestClassifier(verbose=2),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
            }

            params = {
                "Random Forest": {
                    "class_weight": ["balanced"],
                    'n_estimators': [20, 50],
                    'max_depth': [10, 8],
                    'min_samples_split': [2, 5],
                },
                "Decision Tree": {
                    "class_weight": ["balanced"],
                    "criterion": ['gini', "entropy"],
                    "splitter": ['best', 'random'],
                    "max_depth": [3, 4],
                    "min_samples_split": [2, 3],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["auto", "sqrt"]
                },
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=model, params=params)

            # To get the best model from our report Dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name}, Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best model found, Model Name is {best_model_name}, accuracy Score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.train_model_file_apth,
                        obj=best_model
                        )

        except Exception as e:
            raise AppException(e, sys)
