from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pkgs.commons import (
    patient_id_key_literal, timestamp_key_literal, meld_score_key_literal,
    get_input_path, generalize_ratio, real_data_ratio
)
from pkgs.data.harvest import harvest_data_with_interpolate
from pkgs.data.plot import plot_data


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
    num_patients = 800
    data = []

    for patient_id in range(1, num_patients + 1):
        num_records = 200
        start_time = datetime(2022, 1, 1)
        t = start_time
        for record_id in range(num_records):
            t = t + timedelta(days=1)
            if record_id % 20 == 0:
                t = t + timedelta(days=np.random.randint(1, 1000))
            s = np.random.randint(1, 40)
            data.append([patient_id, t, s])  # patient_id, timestamp, score

    df = pd.DataFrame(data, columns=[patient_id_key_literal, timestamp_key_literal, meld_score_key_literal])

    save_path = get_input_path()
    print(f"Saving test data to {save_path}")
    df.to_csv(save_path, index=False)

    print(f"Dataframe with {len(df)} records saved to {save_path}")


def generate_harvested_data_graph(
        real_data_ratio_l: float, generalize_ratio_l: float, num_observed_l, num_predicted_l
):
    df = pd.read_csv(get_input_path())
    train, test, generalize = harvest_data_with_interpolate(
        df, num_observed_l + num_predicted_l, real_data_ratio_l, generalize_ratio_l, interpolate_amount="d"
    )
    # first feature is MELD, second feature is timestamp (as float)
    plot_data(
        np.squeeze(train[:, :, :1], axis=-1),
        np.squeeze(test[:, :, :1], axis=-1),
        np.squeeze(generalize[:, :, :1], axis=-1),
        num_observed_l, num_predicted_l,
        f"window_size{num_observed_l + num_predicted_l}_real_data_ratio{real_data_ratio_l}_generalize_ratio{generalize_ratio_l}")


if __name__ == "__main__":
    pass