from models.commons import get_input_path
from models.data.interpolate import interpolate
import pandas as pd
import numpy as np


# with_manual = True to read the data from INPUT_PATH and interpolate it.
def test_interpolate(with_manual=False):
    # manual testing
    # save the interpolated data to verify the correctness of the interpolate function
    if with_manual:
        df = pd.read_csv(get_input_path())
        inter_amount = "W"
        D = interpolate(df, inter_amount, verbal=True)
        D.to_csv("test_interpolate.csv", index=False)

    # unit testing
    """
    Test the interpolate function using table-driven testing.
    """
    test_cases = [
        {
            "name": "Daily interpolation with gaps",
            "data": {
                "patient_id": ['1', '1', '1', '2', '2'],
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-04 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-03 00:00:00"
                ],
                "score": [10, 20, 30, 15, 25]
            },
            "inter_amount": "d",
            "expected": pd.DataFrame(data=
            {
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00", "2023-01-04 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00"
                ],
                "patient_id": ['1', '1', '1', '1', '2', '2', '2'],
                "score": [10, 20, 20, 30, 15, 15, 25],
                "is_original": [True, True, False, True, True, False, True]
            }
            ),
        },
        {
            "name": "Weekly interpolation with no gaps",
            "data": {
                "patient_id": ['1', '1', '1', '2', '2'],
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00", "2023-01-15 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00"
                ],
                "score": [10, 20, 30, 15, 25]
            },
            "inter_amount": "W",
            "expected_len": 5,
            "expected_len_original": 5,
            "expected": pd.DataFrame(data=
            {
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00", "2023-01-15 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00"
                ],
                "patient_id": ['1', '1', '1', '2', '2'],
                "score": [10, 20, 30, 15, 25],
                "is_original": [True, True, True, True, True]
            }
            ),
        },
        {
            "name": "Weekly interpolation with gaps",
            "data": {
                "patient_id": ['1', '1', '1', '2', '2'],
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00", "2024-01-27 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-15 00:00:00"
                ],
                "score": [10, 20, 30, 15, 25]
            },
            "inter_amount": "W",
            "expected_len": 8,  # limit the forward fill to 1
            "expected_len_original": 5,
            # record at 2024-01-27 should not get omitted even when the forward fill starts from 2023-01-01
            "expected": pd.DataFrame(data=
            {
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00", "2023-01-15 00:00:00", "2024-01-27 00:00:00",
                    "2024-01-28 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00", "2023-01-15 00:00:00"
                ],
                "patient_id": ['1', '1', '1', '1', '1', '2', '2', '2'],
                "score": [10, 20, 20, 30, 30, 15, 15, 25],
                "is_original": [True, True, False, True, False, True, False, True]
            }
            ),
        }
    ]

    for test in test_cases:
        df = pd.DataFrame(test["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        result = interpolate(df, test["inter_amount"])

        try:
            test["expected"]["timestamp"] = pd.to_datetime(test["expected"]["timestamp"])
            test["expected"]["score"] = test["expected"]["score"].astype(float)
            pd.testing.assert_frame_equal(result, test["expected"])
            print(f"Test {test['name']} passed.")
        except AssertionError as e:
            msg = f"Test {test['name']} failed. Interpolated data:\n{result}\nExpected:\n{test['expected']}"
            print(msg)
            raise e

    print("All tests passed.")