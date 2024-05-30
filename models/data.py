import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from models.commons import INPUT_PATH, PATIENT_ID_KEY_LITERAL, TIMESTAMP_KEY_LITERAL, MELD_SCORE_KEY_LITERAL, IS_ORIGINAL_KEY_LITERAL

def generate_test_data():
    """
    Generate test data for patients" scores and save it to a CSV file.

    This function creates a directory "./data" if it doesn"t exist, generates data for 800 patients,
    and saves the data to a CSV file named "patient_scores.csv" in the "./data" folder. Each patient
    will have 100 records with timestamps and scores. The timestamps are spaced one day apart, but 
    every 5th record will have a random gap of 1 to 1000 days added. This gap represents how sparse the real data might be.

    The generated DataFrame will have three columns:
    - "patient_id": An integer ID for each patient.
    - "timestamp": A datetime object representing the timestamp of the record.
    - "score": An integer score between 1 and 40.
    
    Output:
    A CSV file "patient_scores.csv" saved in the "./data" directory.
    """

    os.makedirs("./data", exist_ok=True)

    num_patients = 800
    data = []

    for patient_id in range(1, num_patients + 1):
        num_records = 100
        start_time = datetime(2022, 1, 1)
        t = start_time
        for record_id in range(num_records):
            t = t + timedelta(days=1)
            if record_id % 5 == 0:
                t = t + timedelta(days=np.random.randint(1, 1000))
            s = np.random.randint(1, 41)
            data.append([patient_id, t, s]) # patient_id, timestamp, score

    df = pd.DataFrame(data, columns=[PATIENT_ID_KEY_LITERAL, TIMESTAMP_KEY_LITERAL, MELD_SCORE_KEY_LITERAL])
    df.to_csv(INPUT_PATH, index=False)

    print(f"Dataframe with {len(df)} records saved to {INPUT_PATH}")

def interpolate(df: pd.DataFrame, inter_amount: str, verbal=False) -> pd.DataFrame:
    """
    Interpolate the DataFrame by filling missing values within a specified time interval.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing columns "patient_id", "timestamp", and "score".
    inter_amount (str): The time interval for resampling. "w" for weekly, "d" for daily.

    Returns:
    pd.DataFrame: The interpolated DataFrame with an additional "is_original" column.
    """
    def time_in_original(row_df1, df2):
        """
        Check if the row in df1 is an original row in df2 by comparing the timestamps.
        
        Parameters:
        row_df1 (pd.Series): A row from the first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

        Returns:
        bool: True if there is a corresponding row in df2, False otherwise.
        """
        return row_df1[TIMESTAMP_KEY_LITERAL] in df2[TIMESTAMP_KEY_LITERAL].values

    total_rec = 0
    total_rec_after_interpolation = 0

    # Limit the forward fill because if the data is too sparse, we don't want to fill too many values.
    # We will filter most of the unoriginal data out in later data processing steps.
    ffill_limit = 1 

    df_new = pd.DataFrame(columns=df.columns)

    for _, g in df.groupby(PATIENT_ID_KEY_LITERAL):
        g[TIMESTAMP_KEY_LITERAL] = pd.to_datetime(g[TIMESTAMP_KEY_LITERAL], format="mixed")
        g.set_index(TIMESTAMP_KEY_LITERAL, inplace=True)

        total_rec += len(g)

        g_interpolated = g.resample(inter_amount).ffill(limit=ffill_limit).reset_index()
        g = g.reset_index()

        # concat is needed to keep the original data. otherwise, the resample+ffill will drop some of the original data
        # for example, for timestamps ["2023-01-01 00:00:00", "2023-01-08 00:00:00", "2024-01-27 00:00:00"]
        # resample+ffill with "W" will drop the record at "2024-01-27 00:00:00" because it does not fall on the weekly interval starting from "2023-01-01 00:00:00"
        # see test case "Weekly interpolation with gaps"
        g_interpolated = pd.concat([g_interpolated, g]).sort_values(TIMESTAMP_KEY_LITERAL).drop_duplicates()

        total_rec_after_interpolation += len(g_interpolated)

        # MELD axis=1
        # Check if the interpolated data is original
        g_interpolated[IS_ORIGINAL_KEY_LITERAL] = g_interpolated.apply(lambda row: time_in_original(row, g), axis=1)
        g_interpolated.dropna(subset=[MELD_SCORE_KEY_LITERAL], inplace=True)

        # this handling of empty DataFrames is needed in pandas
        df_new = (
            g_interpolated.copy() if df_new.empty else df_new.copy() if g_interpolated.empty else pd.concat([g_interpolated, df_new])
        )
    
    # need to reset index because we processed the data by patient
    df_new.sort_values(by=['patient_id', 'timestamp'], inplace=True)
    df_new.reset_index(drop=True, inplace=True)
    
    # meld score should be float
    df_new['score'] = df_new['score'].astype(float)
    
    if verbal:
        print(
            f"Total records: {len(df.index)}, total records after interpolation: {len(df_new.index)}.\n"
            f"Number of interpolated records: {total_rec_after_interpolation}.\n"
            f"Number of dropped interpolated records: {total_rec_after_interpolation - len(df_new.index)}.\n"
            f"We limit the forward fill to {ffill_limit} records. The number of dropped interpolated records represents how sparse the data is."
        )

    return df_new

# we have to separate train and test data since the sequence of data is not long enough.
def find_train_test_subarr_interpolated(arr, window_size, min_original_ratio, arr_is_original):
    # print(f'allowed count \n{int(window_size * min_original_ratio)}')
    if len(arr) < window_size:
        return np.array([]), np.array([])
    ans_train, ans_test, original_data_cnt_in_window = [], [], 0

    i = len(arr) - 1
    for _ in range (0, window_size - 1):
        if arr_is_original[i]:
            original_data_cnt_in_window += 1
        i -= 1

    # print(i, ' ', original_data_cnt_in_window)

    searching_test = True
    while i >= 0 and searching_test:
        if arr_is_original[i]:
            original_data_cnt_in_window += 1
        # Only 1 value used for test data. It must be the last one (so that we don't train based on "future" data).
        if original_data_cnt_in_window == window_size:  # test data is not interpolated
            ans_test += [arr[i: i + window_size]]
            searching_test = False
        i -= 1
        if arr_is_original[i + window_size]:
            original_data_cnt_in_window -= 1
    # print(i, ' ', ans_test)

    original_data_cnt_in_window = 0
    for _ in range (0, window_size - 1):
        if arr_is_original[i]:
            original_data_cnt_in_window += 1
        i -= 1

    # print(i, ' ', original_data_cnt_in_window)

    while i >= 0:
        if arr_is_original[i]:
            original_data_cnt_in_window += 1
        if original_data_cnt_in_window >= int(window_size * min_original_ratio):
            ans_train += [arr[i: i + window_size]]
        i -= 1
        if arr_is_original[i + window_size]:
            original_data_cnt_in_window -= 1

    if not ans_test:
        return np.array(ans_train), np.array([])

    return np.array(ans_train[::-1]), np.array([ans_test[-1]])

if __name__ == "__main__":
    generate_test_data()