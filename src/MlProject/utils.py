import os
import sys
from src.MlProject.exception import CustomException
from src.MlProject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

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

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            parameters = param.get(model_name, {})

            gs = GridSearchCV(model, parameters, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_test_pred = best_model.predict(X_test)

            test_model_score = accuracy_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)