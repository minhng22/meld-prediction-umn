import numpy as np
from sklearn.preprocessing import MinMaxScaler

NUM_OBS = 14
NUM_PRED = 14
INPUT_PATH = "./data/meld2_080923.csv"

def get_input_and_validate_data(
        D: np.array, num_observed: int, num_predict: int):
    print("shape of data", D.shape)

    if len(D.shape) != 2:
        raise Exception(
            "input for multi-variable time series forecasting must be 2D np array")

    if D.shape[1] != num_observed + num_predict:
        raise Exception("invalid length of data")

    ip, target = np.array_split(D, [num_observed], 1)

    ip = np.reshape(ip, (ip.shape[0], ip.shape[1], 1))
    target = np.reshape(target, (target.shape[0], target.shape[1], 1))

    print("shape of ip data after split ", ip.shape)
    print("shape of target data after split ", target.shape)

    return ip, target
