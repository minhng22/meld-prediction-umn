import numpy as np
from pkgs.data.commons import bottom_patient_by_len_record, generate_timestep_for_plot, mean_day
import pandas as pd


def test_bottom():
    test_cases = [
        {
            "name": "Test 1 - Basic functionality",
            "input": {
                "d": {
                    "patient1": np.array([[1, 2], [3, 4]]),
                    "patient2": np.array([[1, 2]]),
                    "patient3": np.array([[1, 2], [3, 4], [5, 6]]),
                },
                "bottom_ratio": 0.5
            },
            "expected": ["patient2", "patient1"]
        },
        {
            "name": "Test 2 - All patients have same length records",
            "input": {
                "d": {
                    "patient1": np.array([[1, 2], [3, 4]]),
                    "patient2": np.array([[5, 6], [7, 8]]),
                    "patient3": np.array([[9, 10], [11, 12]]),
                },
                "bottom_ratio": 0.3
            },
            "expected": ["patient1"]
        },
        {
            "name": "Test 3 - bottom_ratio results in fractional index",
            "input": {
                "d": {
                    "patient1": np.array([[1, 2]]),
                    "patient2": np.array([[3, 4], [5, 6]]),
                    "patient3": np.array([[7, 8], [9, 10], [11, 12]]),
                },
                "bottom_ratio": 0.67
            },
            "expected": ["patient1", "patient2", "patient3"] # note that we round up.
        },
        {
            "name": "Test 4 - bottom_ratio is 1 (return all)",
            "input": {
                "d": {
                    "patient1": np.array([[1, 2]]),
                    "patient2": np.array([[3, 4]]),
                    "patient3": np.array([[5, 6]]),
                },
                "bottom_ratio": 1.0
            },
            "expected": ["patient1", "patient2", "patient3"]
        },
        {
            "name": "Test 5 - bottom_ratio is 0 (return none)",
            "input": {
                "d": {
                    "patient1": np.array([[1, 2], [3, 4]]),
                    "patient2": np.array([[5, 6], [7, 8]]),
                    "patient3": np.array([[9, 10], [11, 12]]),
                },
                "bottom_ratio": 0.0
            },
            "expected": []
        }
    ]

    for test in test_cases:
        d = test["input"]["d"]
        bottom_ratio = test["input"]["bottom_ratio"]
        expected = test["expected"]

        try:
            result = bottom_patient_by_len_record(d, bottom_ratio)
            assert result == expected, f"Test {test['name']} failed: expected {expected}, got {result}"
            print(f"Test {test['name']} passed.")
        except AssertionError as e:
            print(f"Test {test['name']} failed.\n")
            raise e


def test_generate_data():
    test_cases = [
        {
            "name": "Test Case 1",
            "X": 3,
            "y": 4,
            "expected": np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        },
        {
            "name": "Test Case 2",
            "X": 2,
            "y": 3,
            "expected": np.array([[1, 2, 3], [1, 2, 3]])
        },
        {
            "name": "Test Case 3",
            "X": 5,
            "y": 1,
            "expected": np.array([[1], [1], [1], [1], [1]])
        },
        {
            "name": "Test Case 4",
            "X": 1,
            "y": 5,
            "expected": np.array([[1, 2, 3, 4, 5]])
        },
    ]

    for test in test_cases:
        X = test["X"]
        y = test["y"]
        expected = test["expected"]

        result = generate_timestep_for_plot(X, y)

        try:
            assert np.array_equal(result, expected), f"Expected {expected} but got {result}"
            assert result.shape == (X, y), f"Expected shape ({X}, {y}) but got {result.shape}"
        except AssertionError as e:
            print(f"Test {test['name']} failed.\n")
            raise e

    print("All tests passed.")


def test_mean_day():
    test_cases = [
        {
            "name": "Test Case 1 - Basic functionality",
            "input": pd.DataFrame({
                "patient_id": [1, 1, 1, 2, 2],
                "timestamp": [
                    "2023-01-01 12:00:00",
                    "2023-01-01 14:00:00",
                    "2023-01-02 09:00:00",
                    "2023-01-01 11:00:00",
                    "2023-01-01 13:00:00"
                ],
                "score": [10, 20, 30, 40, 50]
            }),
            "expected": pd.DataFrame({
                "patient_id": [1, 1, 2],
                "timestamp": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-01"
                ],
                "score": [15.0, 30.0, 45.0]
            }).assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))
        },
        {
            "name": "Test Case 2 - All scores are the same day",
            "input": pd.DataFrame({
                "patient_id": [1, 1, 1],
                "timestamp": [
                    "2023-01-01 12:00:00",
                    "2023-01-01 14:00:00",
                    "2023-01-01 16:00:00"
                ],
                "score": [10, 20, 30]
            }),
            "expected": pd.DataFrame({
                "patient_id": [1],
                "timestamp": [
                    "2023-01-01"
                ],
                "score": [20.0]
            }).assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))
        },
        {
            "name": "Test Case 3 - Different patients, same day",
            "input": pd.DataFrame({
                "patient_id": [1, 2, 1, 2],
                "timestamp": [
                    "2023-01-01 12:00:00",
                    "2023-01-01 14:00:00",
                    "2023-01-02 12:00:00",
                    "2023-01-02 14:00:00"
                ],
                "score": [10, 20, 30, 40]
            }),
            "expected": pd.DataFrame({
                "patient_id": [1, 1, 2, 2],
                "timestamp": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-01",
                    "2023-01-02"
                ],
                "score": [10.0, 30.0, 20.0, 40.0]
            }).assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))
        }
    ]

    for test in test_cases:
        input_df = test["input"]
        expected_df = test["expected"]

        result_df = mean_day(input_df)

        try:
            pd.testing.assert_frame_equal(result_df, expected_df, check_like=True)
        except AssertionError as e:
            print(
                f"Test {test['name']} failed.\n"
                f"Expected:\n{expected_df}\n"
                f"Got:\n{result_df}\n"
            )
            raise e

    print("All tests passed.")