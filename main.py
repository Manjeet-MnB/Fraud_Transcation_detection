import os, sys
import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from src.logger.log import logging
from src.exception.exception import AppException
#from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import CustomData,PredictPipleline


application =  Flask(__name__)

app = application

@app.route('/')
def home_age():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            step=float(request.form.get('step')),
            amount = float(request.form.get('amount')),
            newbalanceOrig = float(request.form.get('newbalanceOrig')),
            newbalanceDest = float(request.form.get('newbalanceDest')),
            isFlaggedFraud = float(request.form.get('isFlaggedFraud'))
            
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipleline()
        pred=predict_pipeline.predict(final_new_data)

        result =pred
        
        if result == 0:
            return render_template("results.html", final_result = "Transaction is not a Fraud:{}".format(result) )
        
        elif result == 1:
        

            return render_template("results.html", final_result = "Transaction is fraud:{}".format(result) )
        
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

