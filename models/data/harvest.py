import numpy as np
import pandas as pd

from models.data.interpolate import interpolate
from models.data.commons import bottom_patient_by_len_record
from models.commons import real_data_ratio, generalize_ratio


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


# get train, test, and generalize data from the input dataframe.
# data is interpolated to fill in missing values.
# data is then collected using sliding window algorithm.
def harvest_data_with_interpolate(df: pd.DataFrame, window_size: int):
    df = interpolate(df, "d")
    ans_train, ans_test = {}, {}

    for s_id, g in df.groupby("patient_id"):
        arr_train_score, arr_test_score = find_train_test(
            g["score"].values, window_size, real_data_ratio, g["is_original"].values
        )
        arr_train_time, arr_test_time = find_train_test(
            g["timestamp"].values, window_size, real_data_ratio, g["is_original"].values
        )
        arr_train_time, arr_test_time = arr_train_time.astype(float), arr_test_time.astype(float)

        if arr_train_time.shape == (0,) or arr_train_score.shape == (0,) or arr_test_score.shape == (
        0,) or arr_test_time.shape == (0,):
            continue

        assert arr_train_time.shape == arr_train_score.shape, Exception(
            f"{arr_train_time.shape[0]} != {arr_train_score.shape[0]}")
        assert len(arr_train_score.shape) == 2 and len(arr_test_score.shape) == 2, Exception(
            f"{arr_test_score.shape} and {arr_train_score.shape} wrong shape")

        ans_train[s_id] = np.concatenate(
            (np.expand_dims(arr_train_score, axis=-1), np.expand_dims(arr_train_time, axis=-1)), axis=2)
        ans_test[s_id] = np.concatenate(
            (np.expand_dims(arr_test_score, axis=-1), np.expand_dims(arr_test_time, axis=-1)), axis=2)

    print(
        f"number of train patients are: {len(ans_train)}\n"
        f"number of test patients are: {len(ans_test)}"
    )

    print("constructing generalize set")
    bot_patient_ids = bottom_patient_by_len_record(ans_test, generalize_ratio)

    # if interpolate ratio is not 1, we can only get data from ans_test
    ans_generalize = dict(dict((k, ans_test[k]) for k in ans_test if k in bot_patient_ids))
    ans_train = dict((k, ans_train[k]) for k in ans_train if k not in bot_patient_ids)
    ans_test = dict((k, ans_test[k]) for k in ans_test if k not in bot_patient_ids)

    print(
        f"number of generalize patients are: {len(ans_generalize)}\n"
        f"number of train patients are: {len(ans_train)}\n"
        f"number of test patients are: {len(ans_test)}"
    )

    print("converting to np")
    ans_train_np, ans_test_np, ans_generalize_np = None, None, None

    for _, v in ans_train.items():
        if ans_train_np is None:
            ans_train_np = v
        else:
            ans_train_np = np.concatenate((ans_train_np, v), axis=0)

    for _, v in ans_test.items():
        if ans_test_np is None:
            ans_test_np = v
        else:
            ans_test_np = np.concatenate((ans_test_np, v), axis=0)

    for _, v in ans_generalize.items():
        if ans_generalize_np is None:
            ans_generalize_np = v
        else:
            ans_generalize_np = np.concatenate((ans_generalize_np, v), axis=0)

    print(
        f"shape of train data is: {ans_train_np.shape}\n"
        f"shape of test data is: {ans_test_np.shape}\n"
        f"shape of generalize data is: {ans_generalize_np.shape}"
    )
    return ans_train_np, ans_test_np, ans_generalize_np
