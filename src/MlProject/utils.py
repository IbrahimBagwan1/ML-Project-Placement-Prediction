import os
import sys
from src.MlProject.exception import CustomException
from src.MlProject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import dill

import pickle
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.MlProject.exception import CustomException

load_dotenv()
password = os.getenv('password')
user = os.getenv('user')
db = os.getenv('db')
host = os.getenv('host')

def read_sql_data():
  logging.info("Reading SQL database started")
  
  try:
    mydb = pymysql.connect(
      user=user,
      password=password,
      host=host,
      db=db
    )
    logging.info("Connection Established", mydb)
    df = pd.read_sql_query("select * from placement_data_full_class", mydb)
    print(df.head())
    return df

  except Exception as e:
    raise CustomException(e)

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open (file_path, "wb") as file_obj:
      pickle.dump(obj, file_obj)

  except Exception as e:
    raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    from sklearn.model_selection import GridSearchCV
    model_report = {}
    best_trained_models = {}

    for model_name, model in models.items():
        param_grid = params[model_name]
        gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        best_trained_models[model_name] = best_model

        y_pred = best_model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        model_report[model_name] = score

    return model_report, best_trained_models


def load_object(file_path):
  try:
    with open(file_path, 'rb') as file_obj:
      return dill.load(file_obj)
  
  except Exception as e:
    raise CustomException(e,sys)
    