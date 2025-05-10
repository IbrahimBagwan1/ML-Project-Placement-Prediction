import sys
import pandas as pd
from src.MlProject.exception import CustomException
from src.MlProject.utils import load_object

class PredictPipeline:
  def __init__(self):
    pass

  def predict(self, features):
    try:
      model_path = 'artifacts/model.pkl'
      preprocessor_path = 'artifacts/preprocessor.pkl'
      model = load_object(file_path = model_path)
      preprocessor = load_object(file_path=preprocessor_path)

      data_scaled = preprocessor.transform(features)
      preds = model.predict(data_scaled)
      return preds
      
    except Exception as e:
      raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 ssc_b: str,
                 hsc_b: str,
                 hsc_s: str,
                 degree_t: str,
                 workex: str,
                 ssc_p: float,
                 hsc_p: float,
                 degree_p: float,
                 etest_p: float):
        self.gender = gender
        self.ssc_b = ssc_b
        self.hsc_b = hsc_b
        self.hsc_s = hsc_s
        self.degree_t = degree_t
        self.workex = workex
        self.ssc_p = ssc_p
        self.hsc_p = hsc_p
        self.degree_p = degree_p
        self.etest_p = etest_p

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "ssc_b": [self.ssc_b],
                "hsc_b": [self.hsc_b],
                "hsc_s": [self.hsc_s],
                "degree_t": [self.degree_t],
                "workex": [self.workex],
                "ssc_p": [self.ssc_p],
                "hsc_p": [self.hsc_p],
                "degree_p": [self.degree_p],
                "etest_p": [self.etest_p]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
