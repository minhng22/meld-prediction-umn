import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_error


def bottom_patient_by_len_record(d: dict, bottom_ratio):
    keys = list(d.keys())
    keys.sort(key=lambda x: d[x].shape[0])

    print("keys", keys)

    return keys[: math.ceil(len(keys) * bottom_ratio)]


def generate_timestep_for_plot(x, y):
    """
    Generates data T of shape (X, y) such that T[:, i] looks like [i+1, i+1, ..., i+1].

    Parameters:
    X (int): Number of rows.
    y (int): Number of columns.

    Returns:
    np.array: The generated data of shape (X, y).
    """
    T = np.zeros((x, y))
    for i in range(y):
        T[:, i] = i + 1
    return T


def split_and_convert_to_3d(data: np.array, num_observed: int, num_predict: int):
    print("shape of data", data.shape)

    if len(data.shape) != 2:
        raise Exception(
            "input for multi-variable time series forecasting must be 2D np array")

    if data.shape[1] != num_observed + num_predict:
        raise Exception("invalid length of data")

    ip, target = np.array_split(data, [num_observed], 1)

    ip = np.reshape(ip, (ip.shape[0], ip.shape[1], 1))
    target = np.reshape(target, (target.shape[0], target.shape[1], 1))

    print("shape of ip data after split ", ip.shape)
    print("shape of target data after split ", target.shape)

    return ip, target


def mean_day(df: pd.DataFrame):
    # Convert 'timestamp' to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    # Set time to 00:00:00
    df["timestamp"] = df["timestamp"].dt.normalize()

    # Group the dataframe by 'patient_id' and 'timestamp', and take the mean of 'score'
    df_mean = (
        df.groupby(["patient_id", pd.Grouper(
            key="timestamp", freq="D")])["score"].mean().reset_index()
    )

    # check if timestamp is unique for each patient_id
    for s_id, g in df_mean.groupby("patient_id"):
        if not g["timestamp"].is_unique:
            print("patient_id: ", s_id)
            print(g)
            print(g["timestamp"].duplicated(keep=False))
            raise Exception("timestamp is not unique for each patient_id")

    return df_mean


def inverse_scale_ops(ips, ops, sc, num_obs, num_pred):
    ops = np.reshape(ops, (-1, num_pred))
    ips = np.reshape(ips, (-1, num_obs))

    tests_ops = np.concatenate((ips, ops), axis=1)

    return sc.inverse_transform(tests_ops)


def calculate_rmse_of_time_step(ip, op):
    rmses = []
    for i in range(ip.shape[1]):
        ip_i = ip[:, i]
        op_i = op[:, i]
        rmse = mean_squared_error(ip_i, op_i, squared=False)
        rmses.append(round(rmse, 3))
    return rmses