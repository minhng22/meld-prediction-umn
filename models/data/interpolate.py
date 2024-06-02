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


if __name__ == "__main__":
    pass
