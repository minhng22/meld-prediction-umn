import joblib
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_fitted_scaler(path):
    with open(path, 'r') as f:
        params = json.load(f)
    
    # We used MinMaxScaler to scale the data
    scaler = MinMaxScaler()
    scaler.fit([[0] * len(params['min'])]) 
    
    scaler.min_ = np.array(params['min'])
    scaler.scale_ = np.array(params['scale'])
    scaler.data_min_ = np.array(params['data_min'])
    scaler.data_max_ = np.array(params['data_max'])
    scaler.data_range_ = np.array(params['data_range'])
    
    return scaler

def save_fitted_scaler(scaler, path):
    joblib.dump(scaler, path)