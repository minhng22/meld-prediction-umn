import numpy as np
import pandas as pd

from pkgs.data.interpolate import interpolate
from pkgs.data.commons import bottom_patient_by_len_record, mean_day


# find_train_test finds the training and test data from the input array.
def find_train_test(arr, window_size, min_original_ratio, arr_is_original):
    if len(arr) < window_size:
        return np.array([]), np.array([])

    def count_originals(start, end):
        return sum(arr_is_original[start:end])

    train_data, test_data, i_test = [], [], 0

    # Identify the test data
    for i in range(len(arr) - window_size, -1, -1):
        if count_originals(i, i + window_size) == window_size:
            test_data.append(arr[i:i + window_size])
            i_test = i
            break

    # Identify the train data
    for i in range(i_test - window_size, -1, -1):
        if count_originals(i, i + window_size) >= int(window_size * min_original_ratio):
            train_data.append(arr[i:i + window_size])

    if not test_data:
        return np.array(train_data), np.array([])

    return np.array(train_data[::-1]), np.array([test_data[-1]])


def get_patients_from_dict_as_np(patients_dict: dict, black_list, in_or_not: bool):
    ans = None
    for k, v in patients_dict.items():
        if (k in black_list) == in_or_not:
            if ans is None:
                ans = v
            else:
                ans = np.concatenate((ans, v), axis=0)
    return ans if ans is not None else np.array([])

# get train, test, and generalize data from the input dataframe.
# data is interpolated to fill in missing values.
# data is then collected using sliding window algorithm.
def harvest_data_with_interpolate(df: pd.DataFrame, window_size: int, real_data_ratio: float, generalize_ratio: float,
                                  interpolate_amount: str):
    df = mean_day(df)
    df = interpolate(df, interpolate_amount, verbal=True)
    ans_train, ans_test = {}, {}

    for s_id, g in df.groupby("patient_id"):
        arr_train_score, arr_test_score = find_train_test(
            g["score"].values, window_size, real_data_ratio, g["is_original"].values
        )
        arr_train_time, arr_test_time = find_train_test(
            g["timestamp"].values, window_size, real_data_ratio, g["is_original"].values
        )

        def add_dim(arr):
            arr = arr.astype(float)
            return np.expand_dims(arr, axis=-1)

        arr_train_time, arr_test_time, arr_train_score, arr_test_score = (
            add_dim(arr_train_time), add_dim(arr_test_time), add_dim(arr_train_score), add_dim(arr_test_score))

        if (arr_train_time.shape[0] == 0 or arr_train_score.shape[0] == 0 or
                arr_test_score.shape[0] == 0 or arr_test_time.shape[0] == 0):
            continue

        ans_train[s_id] = np.concatenate((arr_train_score, arr_train_time), axis=2)
        ans_test[s_id] = np.concatenate((arr_test_score, arr_test_time), axis=2)

    print(
        f"number of train patients are: {len(ans_train)}\n"
        f"number of test patients are: {len(ans_test)}"
    )

    print("constructing generalize set")
    bot_patient_ids = bottom_patient_by_len_record(ans_test, generalize_ratio)

    ans_generalize_np = get_patients_from_dict_as_np(ans_test, bot_patient_ids, True)
    ans_train_np = get_patients_from_dict_as_np(ans_train, bot_patient_ids, False)
    ans_test_np = get_patients_from_dict_as_np(ans_test, bot_patient_ids, False)

    print(
        f"shape of train data is: {ans_train_np.shape}\n"
        f"shape of test data is: {ans_test_np.shape}\n"
        f"shape of generalize data is: {ans_generalize_np.shape}"
    )
    return ans_train_np, ans_test_np, ans_generalize_np
