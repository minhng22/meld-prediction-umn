import pandas as pd
import numpy as np
from datetime import timedelta

from utils import NUM_OBS, INPUT_PATH

GENERALIZE_RATIO = 0.25
REAL_DATA_RATIO = 0.9  # set to 0 if filling na with DL model
RANGE_MELD = [float("-inf"), float("inf")]


def mean_day(df: pd.DataFrame):
    # Convert 'timestamp' to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    # Set time to 00:00:00
    df["timestamp"] = df["timestamp"].dt.normalize()

    # Group the dataframe by 'patient_id' and 'timestamp', and take the mean of 'score'
    df_mean = (
        df.groupby(["patient_id", pd.Grouper(
            key="timestamp", freq="D")])["score"]
            .mean()
            .reset_index()
    )

    # check if timestamp is unique for each patient_id
    for s_id, g in df_mean.groupby("patient_id"):
        if not g["timestamp"].is_unique:
            print("patient_id: ", s_id)
            print(g)
            print(g["timestamp"].duplicated(keep=False))
            raise Exception("timestamp is not unique for each patient_id")

    return df_mean


def interpolate(df: pd.DataFrame, inter_amount: str):
    # Function to check if a row in df1 has a corresponding row in df2 within - 3d
    def within_time(row_df1, df2):
        td = timedelta(days=1)
        if inter_amount == "w":
            td = timedelta(days=6)
        return any((row_df1['timestamp'] - td <= t <= row_df1['timestamp']) for t in df2['timestamp'])

    total_rec = 0
    total_rec_after_interpolation = 0

    df_new = pd.DataFrame(columns=df.columns)

    for _, g in df.groupby("patient_id"):
        g['timestamp'] = pd.to_datetime(g['timestamp'], format='mixed')
        g.set_index("timestamp", inplace=True)

        total_rec += len(g)

        # Timestep would be inter_amount apart
        g_interpolated = g.resample(inter_amount).ffill(limit=1)

        g_interpolated = g_interpolated.reset_index()
        g = g.reset_index()

        total_rec_after_interpolation += len(g_interpolated)

        g_interpolated['is_original'] = g_interpolated.apply(lambda row: within_time(row, g), axis=1)
        g_interpolated.dropna(subset=['score'])

        df_new = pd.concat([df_new, g_interpolated])

    print(
        f"ratio of original data is {total_rec / total_rec_after_interpolation}")
    print(
        f"total_rec: {len(df.index)}, total_rec_after_interpolation: {len(df_new.index)}")

    return df_new


def sort_timestamp(df: pd.DataFrame):
    # Convert 'timestamp' column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    def sort_records(group: pd.DataFrame):
        return group.sort_values("timestamp")

    # Group by 'patient_id' and sort each group by 'timestamp'
    df = df.groupby("patient_id").apply(sort_records).reset_index(drop=True)

    return df


def time_as_feature(data):
    times = []
    print(f"Total number of patients: {len(data['patient_id'].unique())}")

    filtered_df = data.groupby("patient_id").filter(
        lambda group: (RANGE_MELD[0] <= round(group["score"]) <= RANGE_MELD[1]).any())
    print(f"Number of patients with a score in range {RANGE_MELD}: {len(filtered_df['patient_id'].unique())}")

    # convert time to relative time
    for _, group in filtered_df.groupby('patient_id'):
        hallmark_i = -1
        for i, score in enumerate(group['score'].values):
            if hallmark_i == -1 and RANGE_MELD[0] <= round(score) <= RANGE_MELD[1]:
                hallmark_i = i
        if hallmark_i == -1:
            continue
        for i in range(len(group)):
            times.append(i - hallmark_i)
    if len(times) != len(filtered_df):
        raise Exception(f"Number of patients not equal. Expected: {len(filtered_df)}, Actual: {len(times)}")
    filtered_df['time'] = times
    filtered_df.to_csv("./data/interpolated_with_time.csv", index=False)


# notice that len of record of patient in (ans_train, ans_test) might be different from window_size
def interpolate_with_sliding_window(df: pd.DataFrame, window_size: int):
    df = interpolate(df, "d")
    ans_train, ans_test = {}, {}

    for s_id, g in df.groupby("patient_id"):
        arr_train_score, arr_test_score = find_train_test_subarr_interpolated(
            g["score"].values, window_size, REAL_DATA_RATIO, g["is_original"].values
        )
        arr_train_time, arr_test_time = find_train_test_subarr_interpolated(
            g["timestamp"].values, window_size, REAL_DATA_RATIO, g["is_original"].values
        )
        arr_train_time, arr_test_time = arr_train_time.astype(float), arr_test_time.astype(float)

        if arr_train_time.shape == (0,) or arr_train_score.shape == (0,) \
                or arr_test_score.shape == (0,) or arr_test_time.shape == (0,):
            continue

        assert arr_train_time.shape == arr_train_score.shape, Exception(
            f"{arr_train_time.shape[0]} != {arr_train_score.shape[0]}")
        assert len(arr_train_score.shape) == 2 and len(arr_test_score.shape) == 2, Exception(
            f"{arr_test_score.shape} and {arr_train_score.shape} wrong shape")

        arr_train = np.concatenate((
            np.reshape(arr_train_score, (arr_train_score.shape[0], arr_train_score.shape[1], 1)),
            np.reshape(arr_train_time, (arr_train_time.shape[0], arr_train_time.shape[1], 1))), axis=2)

        arr_test = np.concatenate((
            np.reshape(arr_test_score, (arr_test_score.shape[0], arr_test_score.shape[1], 1)),
            np.reshape(arr_test_time, (arr_test_time.shape[0], arr_test_time.shape[1], 1))), axis=2)

        # filter by MELD of the last day
        if RANGE_MELD[0] <= arr_test_score[:, NUM_OBS - 1][0] <= RANGE_MELD[1]:
            ans_train[s_id] = arr_train
            ans_test[s_id] = arr_test

    print(f"number of train patients are: {len(ans_train)}")
    print(f"number of test patients are: {len(ans_test)}")

    print("constructing generalize set")
    bot_patient_ids = bottom_patient_by_len_record(ans_test)

    # if interpolate ratio is not 1, we can only get data from ans_test
    ans_generalize = dict(
        dict((k, ans_test[k]) for k in ans_test if k in bot_patient_ids),
    )

    ans_train = dict((k, ans_train[k])
                     for k in ans_train if k not in bot_patient_ids)
    ans_test = dict((k, ans_test[k])
                    for k in ans_test if k not in bot_patient_ids)

    print(f"number of generalize patients are: {len(ans_generalize)}")
    print(f"number of train patients are: {len(ans_train)}")
    print(f"number of test patients are: {len(ans_test)}")

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

    print(f"shape of train data is: {ans_train_np.shape}")
    print(f"shape of test data is: {ans_test_np.shape}")
    print(f"shape of generalize data is: {ans_generalize_np.shape}")

    return ans_train_np, ans_test_np, ans_generalize_np


def split_train_with_reshape(arr: np.ndarray, split_id: int, axis: int):
    if len(arr.shape) != 2:
        raise Exception(f"data shape is not 2, but {arr.shape}")
    arr = np.reshape(arr, (arr.shape[0], arr.shape[1], 1))
    return np.array_split(arr, [split_id], axis=axis)


# we have to separate train and test data since the sequence of data is not long enough.
def find_train_test_subarr_interpolated(
        arr, window_size, min_original_ratio, arr_is_original
):
    #print(f'allowed count \n{int(window_size * min_original_ratio)}')
    if len(arr) < window_size:
        return np.array([]), np.array([])
    ans_train, ans_test, original_data_cnt_in_window = [], [], 0

    i = len(arr) - 1
    for _ in range (0, window_size - 1):
        if arr_is_original[i] == True:
            original_data_cnt_in_window += 1
        i -= 1

    #print(i, ' ', original_data_cnt_in_window)

    searching_test = True
    while i >= 0 and searching_test:
        if arr_is_original[i] == True:
            original_data_cnt_in_window += 1
        # Only 1 value used for test data. It must be the last one (so that we don't train based on "future" data).
        if original_data_cnt_in_window == window_size:  # test data is not interpolated
            ans_test += [arr[i: i + window_size]]
            searching_test = False
        i -= 1
        if arr_is_original[i + window_size] == True:
            original_data_cnt_in_window -= 1
    #print(i, ' ', ans_test)

    original_data_cnt_in_window = 0
    for _ in range (0, window_size - 1):
        if arr_is_original[i] == True:
            original_data_cnt_in_window += 1
        i -= 1

    #print(i, ' ', original_data_cnt_in_window)

    while i >= 0:
        if arr_is_original[i] == True:
            original_data_cnt_in_window += 1
        if original_data_cnt_in_window >= int(window_size * min_original_ratio):
            ans_train += [arr[i: i + window_size]]
        i -= 1
        if arr_is_original[i + window_size] == True:
            original_data_cnt_in_window -= 1

    if not ans_test:
        return np.array(ans_train), np.array([])

    return np.array(ans_train[::-1]), np.array([ans_test[-1]])


def bottom_patient_by_len_record(d):
    keys = list(d.keys())
    keys.sort(key=lambda x: d[x].shape[0])  # short patient by number of record

    return keys[: int(len(list(d.keys())) * GENERALIZE_RATIO)]


def test_interpolate():
    df = pd.read_csv(INPUT_PATH)
    df = interpolate(df, "d")
    df.to_csv("interpolated.csv")


def test_find_train_test_subarr_interpolated():
    data = {
        'score': [
            66, 77, 89, 78, 77, 77, 56, 95, 69, 73,
            74, 80, 57, 73, 71, 60, 87, 95, 87, 77,
            68, 84, 88, 97, 82, 77, 97, 53, 73, 96],
        'is_original': [
            False, True, True, True, True, True, True, True, False, True,
            True, True, False, True, True, False, True, True, True, True,
            True, True, True, True, True, True, True, False, True, True
        ]
    }
    df = pd.DataFrame(data)
    train, test = find_train_test_subarr_interpolated(df['score'].values, 6, 0.9, df['is_original'].values)
    print('train \n', train)
    print('test \n', test)


if __name__ == "__main__":
    test_find_train_test_subarr_interpolated()
