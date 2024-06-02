from models.commons import get_input_path
from models.data.interpolate import interpolate, find_train_test_subarray_interpolated
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
        except AssertionError as e:
            msg = f"Test {test['name']} failed. Interpolated data:\n{result}\nExpected:\n{test['expected']}"
            print(msg)
            raise e

    print("All tests passed.")


def test_find_train_test_subarray_interpolated():
    test_cases = [
        {
            "name": "Simple case with exact match",
            "arr": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "window_size": 4,
            "min_original_ratio": 0.75,
            "arr_is_original": np.array([True, True, True, True, True, True, True, True, True, True]),
            "expected_train": np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
            "expected_test": np.array([[7, 8, 9, 10]])
        },
        {
            "name": "Insufficient original data for training",
            "arr": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "window_size": 4,
            "min_original_ratio": 0.75,
            "arr_is_original": np.array([False, False, False, True, True, True, True, True, True, True]),
            "expected_train": np.array([[3, 4, 5, 6]]),
            "expected_test": np.array([[7, 8, 9, 10]])
        },
        {
            "name": "Mixed original and interpolated data",
            "arr": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "window_size": 4,
            "min_original_ratio": 0.5,
            "arr_is_original": np.array([True, False, True, True, True, False, True, True, False, True]),
            "expected_train": np.array([]),
            "expected_test": np.array([]) # there is no window with 100% original data for test
        },
        {
            "name": "All data is interpolated",
            "arr": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "window_size": 4,
            "min_original_ratio": 0.75,
            "arr_is_original": np.array([False, False, False, False, False, False, False, False, False, False]),
            "expected_train": np.array([]),
            "expected_test": np.array([])
        },
        {
            "name": "Data length smaller than window size",
            "arr": np.array([1, 2, 3]),
            "window_size": 4,
            "min_original_ratio": 0.75,
            "arr_is_original": np.array([True, True, True]),
            "expected_train": np.array([]),
            "expected_test": np.array([])
        }
    ]

    for test in test_cases:
        train, test_arr = find_train_test_subarray_interpolated(test['arr'], test['window_size'], test['min_original_ratio'], test['arr_is_original'])
        try:
            np.testing.assert_array_equal(
                train, test['expected_train'],
                err_msg=f"Train data for test {test['name']} is incorrect. Expected: {test['expected_train']}, got: {train}"
            )
            np.testing.assert_array_equal(
                test_arr, test['expected_test'],
                err_msg=f"Test data for test {test['name']} is incorrect. Expected: {test['expected_test']}, got: {test_arr}"
            )
        except AssertionError as e:
            f"Test {test['name']} failed.\n"
            raise e


if __name__ == "__main__":
    test_interpolate()
    test_find_train_test_subarray_interpolated()