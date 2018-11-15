import sys

from flask import Flask
from flask import json
from flask import request, render_template, flash

from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

import numpy as np
import pickle
import itertools
from collections import OrderedDict

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

PRED_DATA = 'prediction_data.data'
NN_MODEL = 'nn_model.model'
LIN_REG_MODEL = 'lin_reg_model.model'
RAND_FOREST_MODEL = 'rand_forest.model'
TPOT_MODEL = 'tpot_pipeline.model'
AUTOSK_MODEL = 'automl.model'


@app.route('/', methods=['GET'])
def dropdown():
	pdata = pickle.load(open(PRED_DATA, 'rb'))
	return render_template('home.html', algo = ['NEURAL_NETWORK', 'LINEAR_REGRESSION', 'RANDOM_FOREST', 'TPOT_MODEL'],
						   term = pdata[3]['term'], grade = pdata[3]['grade'],sub_grade = pdata[3]['sub_grade'],
						   emp_length = pdata[3]['emp_length'], home_ownership = pdata[3]['home_ownership'],
						   verification_status = pdata[3]['verification_status'], purpose = pdata[3]['purpose'],
						   addr_state = pdata[3]['addr_state'], status = ['NOT APPROVED','APPROVED'])

# sample = {"loan_amnt" : 10902,"term" : " 36 months","installment": 100,"grade": "A","sub_grade": "A2","emp_length": "7 years","home_ownership": "MORTGAGE","annual_inc": 20000,"verification_status": "Source Verified","purpose": "small_business","addr_state": "MA","dti": 15.2,"delinq_2yrs": 1,"inq_last_6mths": 3,"loan_status_Binary": 1}

def get_interest_rate(algo, model, sample, df_max, df_min, categories, features):
    # We will need to map the inputs to a normalized form such that the max and min
    # are in accordnace with the max and min values on which the model was trained.
    # We also need to make sure that the features are in the same order as they were in the training set.

    features_vector=OrderedDict()
    for feat in features:
        features_vector[feat] = sample[feat]
    
    # check numerical features
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    # remove target feature
    df_max_feature =df_max[~df_max.index.isin(['int_rate'])]
    df_min_feature =df_min[~df_min.index.isin(['int_rate'])]
    feature_vector_norm = []

    # convert categorical features
    for key, value in sample.items():
        if key in categories:  		
            features_vector[key] = categories[key].index(features_vector[key])
        features_vector[key] = (float(features_vector[key])-float(df_min_feature[key]))/(float(df_max_feature[key])-float(df_min_feature[key]))
        feature_vector_norm.append(features_vector[key])
    

    if algo =='TPOT_MODEL':
        return model.predict([feature_vector_norm])

    return (model.predict([np.array(feature_vector_norm)]))*(df_max['int_rate']-df_min['int_rate']) + df_min['int_rate']

@app.route('/predict',methods = ['POST'])
def predict():
	try:
		if request.method == 'POST':
			sample = request.form.to_dict()
			algo = sample.pop('algo')
			if algo == 'NEURAL_NETWORK':
				model = pickle.load(open(NN_MODEL, 'rb'))
			elif algo == 'LINEAR_REGRESSION':
				model = pickle.load(open(LIN_REG_MODEL, 'rb'))
			elif algo == 'RANDOM_FOREST':
				model =  pickle.load(open(RAND_FOREST_MODEL, 'rb'))
			elif algo == 'AUTO_SKLEARN_MODEL':
				
				model =  pickle.load(open(AUTOSK_MODEL, 'rb'))
			else:
				model = pickle.load(open(TPOT_MODEL, 'rb'))
			
			pdata = pickle.load(open(PRED_DATA, 'rb'))
			if algo == 'H2O_MODEL':
				
				sample_tup = sample.items()
				print(type(sample_tup))
				return render_template("prediction.html", interest = (str(model.predict(sample_tup))))
			return render_template("prediction.html", interest = (str(get_interest_rate(algo, model, sample, pdata[1], pdata[2], pdata[3],pdata[4]))))
		
	except Exception as ex:
		print(ex)
		return "Please fill in all the fields!"


@app.route('/api/predict', methods = ['POST'])
def api_message():
	if request.headers['Content-Type'] == 'application/json':
		return "Interest rate for "+ str(request.json) +" is " + predict(request.get_json()) + " using .\n" 
	else:
		return "415 Unsupported Media Type"

if __name__== '__main__':
  	app.run(debug=False,host='0.0.0.0', port=5000) 
