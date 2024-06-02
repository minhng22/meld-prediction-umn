import pandas as pd
import numpy as np

from models.commons import PATIENT_ID_KEY_LITERAL, TIMESTAMP_KEY_LITERAL, MELD_SCORE_KEY_LITERAL, \
    IS_ORIGINAL_KEY_LITERAL


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
            g_interpolated.copy() if df_new.empty else df_new.copy() if g_interpolated.empty else pd.concat(
                [g_interpolated, df_new])
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


def find_train_test_subarray_interpolated(arr, window_size, min_original_ratio, arr_is_original):
    if len(arr) < window_size:
        return np.array([]), np.array([])
    ans_train, ans_test, original_data_cnt_in_window = [], [], 0

    # finding the test data
    i = len(arr) - 1
    for _ in range(0, window_size - 1):
        if arr_is_original[i]:
            original_data_cnt_in_window += 1
        i -= 1

    searching_test = True
    while i >= 0 and searching_test:
        if arr_is_original[i]:
            original_data_cnt_in_window += 1
        # Only 1 window used for test data. It must be the last one (so that we don't train based on "future" data).
        # test data is not interpolated. We can have interpolated test data and filter interpolated data points
        # while evaluating accuracy but edge case of that is where test data points are all interpolated which
        # makes model accuracy evaluation meaningless.
        if original_data_cnt_in_window == window_size:
            ans_test += [arr[i: i + window_size]]
            searching_test = False
        i -= 1
        if arr_is_original[i + window_size]:
            original_data_cnt_in_window -= 1

    # finding the train data
    original_data_cnt_in_window = 0
    for _ in range(0, window_size - 1):
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
    pass
