import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from datetime import datetime

import numpy as np
from xgboost import XGBRegressor

from pkgs.data.commons import inverse_scale_ops
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.scaler import get_fitted_scaler
from pkgs.commons import meld_scaler_path, time_stamp_scaler_path, xgboost_model_path

supporting_num_obs = [5]
supporting_num_pred = [1, 3, 5, 7]

def main(meld_na_scores, time_stamps, num_of_predicting_days):
    print(f"Predicting MELD scores at {datetime.now()}")
    meld_na_scores, time_stamps, num_obs, num_pred, err = process_input(meld_na_scores, time_stamps, num_of_predicting_days)
    if err is not None:
        return err
    
    meld_na_scores = np.reshape(meld_na_scores, (1, num_obs))
    time_stamps = np.reshape(time_stamps, (1, num_obs))
    print("Data loaded successfully.")

    meld_scaler = get_fitted_scaler(meld_scaler_path(num_obs=num_obs, num_pred=num_pred))
    timestamp_scaler = get_fitted_scaler(time_stamp_scaler_path(num_obs=num_obs, num_pred=num_pred))
    print("Scalers loaded successfully.")

    data = SlidingWindowDataset()
    data.setup_automated_tool(meld_scaler, timestamp_scaler, meld_na_scores, time_stamps, num_obs, num_pred)

    m = XGBRegressor()
    m.load_model(xgboost_model_path(num_obs, num_pred, "xgboost"))
    print("Model loaded successfully.")

    ip = data.get_test_ips()
    num_feature_input = 2  # meld and timestamp
    test_ips_reshaped = np.reshape(ip, (-1, num_obs * num_feature_input))
    model_pred = m.predict(test_ips_reshaped)

    pred_full = inverse_scale_ops(data.get_test_ip_meld(), model_pred, data.meld_sc, num_obs, num_pred)
    pred_future = pred_full[:, num_obs:]
    pred_future = np.round(pred_future, decimals=3)
    pred_future = np.reshape(pred_future, num_pred)

    return f"MELD score in the next {num_pred} days is {pred_future}"


def process_input(meld_na_scores, time_stamps, num_of_predicting_days):
    def parse_float_list(ip):
        try:
            # Remove brackets and split by comma
            cleaned = ip.strip('[]')
            # Split and convert each element to float
            result = [float(x.strip()) for x in cleaned.split(',')]
            return result, None
        except ValueError as e:
            return None, "Invalid format: All elements must be numbers"
        except Exception as e:
            return None, "Invalid string format. Expected format: [num1, num2, ...]"
    
    def parse_str_list(ip):
        try:
            # Remove brackets and split by comma
            cleaned = ip.strip('[]')
            # Split and convert each element to float
            result = [x.replace("'", "") for x in cleaned.split(',')]
            return result, None
        except ValueError as e:
            return None, "Invalid format: All elements must be numbers"
        except Exception as e:
            return None, "Invalid string format. Expected format: [num1, num2, ...]"
        
    print("Processing input")
    print(f"meld_na_scores: {meld_na_scores}, time_stamps: {time_stamps}, num_of_predicting_days: {num_of_predicting_days}")

    meld_na_scores, err = parse_float_list(meld_na_scores)
    if err is not None:
        print('Error parsing meld_na_scores:', err)
        return None, None, None, None, f'Error parsing meld_na_scores: {err}'
    if (len(meld_na_scores) not in supporting_num_obs):
        print("Please provide a valid list of numbers for MELDNa scores.")
        return None, None, None, None, "Please provide a valid list of numbers for MELDNa scores."

    time_stamps, err = parse_str_list(time_stamps)
    if len(time_stamps) not in supporting_num_obs:
        print("Please provide a valid list of date times for time_stamps.")
        return None, None, None, None, "Please provide a valid list of date times for time_stamps."

    # Convert time_stamps to datetime objects
    try:
        # Convert to datetime64 first
        time_stamps = np.array([np.datetime64(ts) for ts in time_stamps])
        
        # Convert to numeric values for comparison
        time_nums = time_stamps.astype('datetime64[ns]').astype('int64')
        
        # Check if sorted
        if not np.all(np.diff(time_nums) > 0):
            raise ValueError("Timestamps are not sorted in ascending order")
            
        # Check if evenly spaced
        time_diffs = np.diff(time_nums)
        if not np.allclose(time_diffs, time_diffs[0], rtol=1e-10):
            raise ValueError("Timestamps are not evenly spaced")
            
    except ValueError as e:
        print(f"Error with time_stamps: {e}")
        return None, None, None, None, f"Error with time_stamps: {e.args[0]}"

    num_pred = int(num_of_predicting_days)
    if num_pred not in supporting_num_pred:
        print("Please provide a valid number of predicting MELD.")
        return None, None, None, None, "Please provide a valid number of predicting MELD."

    if len(time_stamps) != len(meld_na_scores):
        print("Number of time_stamps does not match number of meld_na_scores.")
        return None, None, None, None, "Number of time_stamps does not match number of meld_na_scores."

    meld_na_scores = np.array(meld_na_scores)
    time_stamps = np.array(time_stamps).astype(float)

    return meld_na_scores, time_stamps, len(meld_na_scores), num_pred, None


if __name__ == '__main__':
    main()
