import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.MlProject.exception import CustomException
from src.MlProject.logger import logging
import os

from src.MlProject.utils import save_object

@dataclass
class DataTransformationConfig: 
  preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()
  
  def get_data_transformer_object(self):
    '''
    This function is responsible for data transformation for the placement dataset
    '''
    try:
        numerical_features = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p']
        categorical_features = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex']

        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ('scaler', StandardScaler(with_mean=False))
        ])
        logging.info(f"Categorical Columns: {categorical_features}")
        logging.info(f"Numerical Columns: {numerical_features}")
        
        preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_features),
            ('cat_pipeline', cat_pipeline, categorical_features)
        ])
        return preprocessor
    
    except Exception as e:
        raise CustomException(e, sys)
  
  def initiate_data_transformation(self, train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Reading the Train and Test data")

        preprocessing_obj = self.get_data_transformer_object()
        target_column = "status"

        X_train = train_df.drop(columns=[target_column], axis=1)
        y_train = train_df[target_column].map({'Placed': 1, 'Not Placed': 0})
        X_test = test_df.drop(columns=[target_column], axis=1)
        y_test = test_df[target_column].map({'Placed': 1, 'Not Placed': 0})
        logging.info("Applying preprocessing on training and testing data")

        X_train_transformed = preprocessing_obj.fit_transform(X_train)
        X_test_transformed = preprocessing_obj.transform(X_test)
        train_arr = np.c_[X_train_transformed, y_train]
        test_arr = np.c_[X_test_transformed, y_test]

        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )
        logging.info("Preprocessing object saved")

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )
    
    except Exception as e:
        raise CustomException(e, sys)
