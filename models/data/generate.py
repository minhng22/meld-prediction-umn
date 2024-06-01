import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from models.commons import (
    PATIENT_ID_KEY_LITERAL, TIMESTAMP_KEY_LITERAL, MELD_SCORE_KEY_LITERAL,
    get_input_path
)


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
        num_records = 100
        start_time = datetime(2022, 1, 1)
        t = start_time
        for record_id in range(num_records):
            t = t + timedelta(days=1)
            if record_id % 5 == 0:
                t = t + timedelta(days=np.random.randint(1, 1000))
            s = np.random.randint(1, 41)
            data.append([patient_id, t, s])  # patient_id, timestamp, score

    df = pd.DataFrame(data, columns=[PATIENT_ID_KEY_LITERAL, TIMESTAMP_KEY_LITERAL, MELD_SCORE_KEY_LITERAL])

    save_path = get_input_path()
    print(f"Saving test data to {save_path}")
    df.to_csv(save_path, index=False)

    print(f"Dataframe with {len(df)} records saved to {save_path}")


if __name__ == "__main__":
    generate_test_data()