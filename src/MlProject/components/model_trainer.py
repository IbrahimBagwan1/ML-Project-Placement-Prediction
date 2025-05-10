import sys
import os
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.MlProject.exception import CustomException
from src.MlProject.logger import logging
from src.MlProject.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
  def __init__(self):
      self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_arr, test_arr):
    try:
        logging.info("Splitting train and test input data")
        X_train, y_train, X_test, y_test = (
            train_arr[:, :-1],
            train_arr[:, -1],
            test_arr[:, :-1],
            test_arr[:, -1]
        )

        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "CatBoost": CatBoostClassifier(verbose=False),
            "AdaBoost": AdaBoostClassifier(),
            "KNN": KNeighborsClassifier()
        }

        params = {
            "Logistic Regression": {
                "C": [0.1, 1.0, 10.0],
                "solver": ['liblinear']
            },
            "Decision Tree": {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 10]
            },
            "Random Forest": {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, 20]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            "XGBoost": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            "CatBoost": {
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [50, 100]
            },
            "AdaBoost": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1]
            },
            "KNN": {
                'n_neighbors': [3, 5, 7]
            }
        }

        model_report, trained_models = evaluate_models(X_train, y_train, X_test, y_test, models, params)


        
        best_model_score = max(model_report.values())
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = trained_models[best_model_name]


        logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )

        predictions = best_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        return acc
    
    except Exception as e:
      raise CustomException(e, sys)
