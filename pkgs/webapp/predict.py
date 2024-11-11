import ast
import sys
from datetime import datetime

import numpy as np
from xgboost import XGBRegressor

from pkgs.data.commons import inverse_scale_ops
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.scaler import get_fitted_scaler
from pkgs.commons import meld_scaler_path, time_stamp_scaler_path, xgboost_model_path

supporting_num_obs = [5]
supporting_num_pred = [1, 3, 5, 7]


def main():
    meld_na_scores, time_stamps, num_obs, num_pred, err = process_input()
    if err is not None:
        print(err)
        return
    meld_na_scores = np.reshape(meld_na_scores, (1, num_obs))
    time_stamps = np.reshape(time_stamps, (1, num_obs))

    meld_scaler = get_fitted_scaler(meld_scaler_path(num_obs=num_obs, num_pred=num_pred))
    timestamp_scaler = get_fitted_scaler(time_stamp_scaler_path(num_obs=num_obs, num_pred=num_pred))

    data = SlidingWindowDataset()
    data.setup_automated_tool(meld_scaler, timestamp_scaler, meld_na_scores, time_stamps, num_obs, num_pred)

    m = XGBRegressor()
    m.load_model(xgboost_model_path(num_obs, num_pred, "xgboost"))

    ip = data.get_test_ips()
    num_feature_input = 2  # meld and timestamp
    test_ips_reshaped = np.reshape(ip, (-1, num_obs * num_feature_input))
    model_pred = m.predict(test_ips_reshaped)

    pred_full = inverse_scale_ops(data.get_test_ip_meld(), model_pred, data.meld_sc, num_obs, num_pred)
    pred_future = pred_full[:, num_obs:]
    pred_future = np.round(pred_future, decimals=3)
    pred_future = np.reshape(pred_future, num_pred)

    print(f"MELD score in the next {num_pred} days is {pred_future}")


def process_input():
    meld_na_scores = ast.literal_eval(sys.argv[1])
    if (not isinstance(meld_na_scores, list)
            or not all(isinstance(score, (int, float)) for score in meld_na_scores)
            or len(meld_na_scores) not in supporting_num_obs):
        print("Please provide a valid list of numbers for MELDNa scores.")
        return None, None, None, None, ValueError

    time_stamps = ast.literal_eval(sys.argv[2])
    if not isinstance(time_stamps, list) or len(time_stamps) not in supporting_num_obs:
        print("Please provide a valid list of date times for time_stamps.")  # sfed
        return None, None, None, None, ValueError

    # Convert time_stamps to datetime objects
    try:
        time_stamps = np.array([np.datetime64(ts, 'ns') for ts in time_stamps])
    except ValueError as e:
        print(f"Error converting time_stamps: {e}")
        return None, None, None, None, ValueError

    num_pred = int(ast.literal_eval(sys.argv[3]))
    if num_pred not in supporting_num_pred:
        print("Please provide a valid number of predicting MELD.")
        return None, None, None, None, ValueError

    if len(time_stamps) != len(meld_na_scores):
        print("Number of time_stamps does not match number of meld_na_scores.")
        return None, None, None, None, ValueError

    meld_na_scores = np.array(meld_na_scores)
    time_stamps = np.array(time_stamps).astype(float)

    return meld_na_scores, time_stamps, len(meld_na_scores), num_pred, None


if __name__ == '__main__':
    main()
