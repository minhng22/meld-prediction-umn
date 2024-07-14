import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def get_fitted_scaler(path):
    # Load parameters from JSON
    with open(path, 'r') as f:
        scaler_params = json.load(f)

    # Initialize MinMaxScaler with loaded parameters
    scaler = MinMaxScaler()
    scaler.min_ = np.array(scaler_params['min'])
    scaler.scale_ = np.array(scaler_params['scale'])
    scaler.data_min_ = np.array(scaler_params['data_min'])
    scaler.data_max_ = np.array(scaler_params['data_max'])
    scaler.data_range_ = np.array(scaler_params['data_range'])

    return scaler

def save_fitted_scaler(scaler, path):
    # Extract parameters
    scaler_params = {
        'min': scaler.min_.tolist(),
        'scale': scaler.scale_.tolist(),
        'data_min': scaler.data_min_.tolist(),
        'data_max': scaler.data_max_.tolist(),
        'data_range': scaler.data_range_.tolist()
    }

    # Save parameters to JSON
    with open(path, 'w') as f:
        json.dump(scaler_params, f)