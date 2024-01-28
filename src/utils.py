import numpy as np 
import os, sys
import pandas as pd
import logging
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    



def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models, param):
    model_report = {}
    best_model_accuracy = 0
    best_model = None

    for model_name, model in models.items():
        logging.info(f"Evaluating {model_name}")
        for params in param[model_name]:
            model.set_params(**params)
            scores = cross_val_score(model, X_train, y_train, cv=5)
            accuracy = scores.mean()
            model_report[model_name] = accuracy

            if accuracy > best_model_accuracy:
                best_model_accuracy = accuracy
                best_model = model_name

    model_report['best_model'] = best_model
    model_report['best_accuracy'] = best_model_accuracy

    return model_report


def save_object(file_path, obj):
    with open(file_path, 'wb') as f:
        joblib.dump(obj, f)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

    