import pandas as pd
import numpy as np

from pkgs.commons import patient_id_key_literal, timestamp_key_literal, meld_score_key_literal, \
    is_original_key_literal


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
        return row_df1[timestamp_key_literal] in df2[timestamp_key_literal].values

    total_rec = 0
    total_rec_after_interpolation = 0

    # Limit the forward fill because if the data is too sparse, we don't want to fill too many values.
    # We will filter most of the unoriginal data out in later data processing steps.
    ffill_limit = 100

    df_new = pd.DataFrame(columns=df.columns)

    for _, g in df.groupby(patient_id_key_literal):
        g[timestamp_key_literal] = pd.to_datetime(g[timestamp_key_literal], format="mixed")
        g.set_index(timestamp_key_literal, inplace=True)

        total_rec += len(g)

        g_interpolated = g.resample(inter_amount).ffill(limit=ffill_limit).reset_index()
        g = g.reset_index()

        # concat is needed to keep the original data. otherwise, the resample+ffill will drop some of the original data
        # for example, for timestamps ["2023-01-01 00:00:00", "2023-01-08 00:00:00", "2024-01-27 00:00:00"]
        # resample+ffill with "W" will drop the record at "2024-01-27 00:00:00" because it does not fall on the weekly interval starting from "2023-01-01 00:00:00"
        # see test case "Weekly interpolation with gaps"
        g_interpolated = pd.concat([g_interpolated, g]).sort_values(timestamp_key_literal).drop_duplicates()

        total_rec_after_interpolation += len(g_interpolated)

        # MELD axis=1
        # Check if the interpolated data is original
        g_interpolated[is_original_key_literal] = g_interpolated.apply(lambda row: time_in_original(row, g), axis=1)
        g_interpolated.dropna(subset=[meld_score_key_literal], inplace=True)

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
            f"Total records: {len(df.index)}, total records after processed: {len(df_new.index)}.\n"
            f"Number of records after interpolation: {total_rec_after_interpolation}.\n"
            f"Number of dropped interpolated records: {total_rec_after_interpolation - len(df_new.index)}.\n"
            f"We limit the forward fill to {ffill_limit} records. The number of dropped interpolated records represents how sparse the data is."
        )

    return df_new