from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.MlProject.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            ssc_b=request.form.get('ssc_b'),
            hsc_b=request.form.get('hsc_b'),
            hsc_s=request.form.get('hsc_s'),
            degree_t=request.form.get('degree_t'),
            workex=request.form.get('workex'),
            ssc_p=float(request.form.get('ssc_p')),
            hsc_p=float(request.form.get('hsc_p')),
            degree_p=float(request.form.get('degree_p')),
            etest_p=float(request.form.get('etest_p'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=int(results[0]))

if __name__=="__main__":
  app.run(host="0.0.0.0") 