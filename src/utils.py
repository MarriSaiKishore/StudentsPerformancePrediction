import os
import sys
import numpy as np
import pandas as pd

import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys) 


def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        model_names = list(models.keys())

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = model_names[i]  # Get the model name
            para = param[model_name] 

            gs=GridSearchCV(model,para,cv=3,scoring='r2')
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            
            #model.fit(X_train,y_train)
            ytrainpred=model.predict(X_train)
            ytestpred=model.predict(X_test)

            trainmodelscore=r2_score(y_train,ytrainpred)
            testmodelscore=r2_score(y_test,ytestpred)

            report[list(models.keys())[i]]=testmodelscore
        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    