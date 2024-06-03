import numpy as np
from models.data.harvest import find_train_test, harvest_data_with_interpolate, get_patients_from_dict_as_np
import pandas as pd


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
            "expected_test": np.array([])  # there is no window with 100% original data for test
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
        train, test_arr = find_train_test(test['arr'], test['window_size'], test['min_original_ratio'],
                                          test['arr_is_original'])
        try:
            np.testing.assert_array_equal(
                train, test['expected_train'],
                err_msg=f"Train data for test {test['name']} is incorrect. Expected: {test['expected_train']}, got: {train}"
            )
            np.testing.assert_array_equal(
                test_arr, test['expected_test'],
                err_msg=f"Test data for test {test['name']} is incorrect. Expected: {test['expected_test']}, got: {test_arr}"
            )
            print(f"Test {test['name']} passed.")
        except AssertionError as e:
            f"Test {test['name']} failed.\n"
            raise e


def test_harvest_data_with_interpolate():
    test_cases = [
        {
            "name": "Test 1 - Basic functionality",
            "input": {
                "df": pd.DataFrame({
                    "patient_id": ['1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '3', '3', '3', '3', '3'],
                    "timestamp": [
                        "2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-05 00:00:00", "2023-01-06 00:00:00",
                        "2023-01-07 00:00:00",
                        "2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-05 00:00:00", "2023-01-06 00:00:00",
                        "2023-01-07 00:00:00",
                        "2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-05 00:00:00", "2023-01-06 00:00:00",
                        "2023-01-07 00:00:00",
                    ],
                    "score": [10, 20, 30, 15, 25, 10, 20, 30, 15, 25, 10, 20, 30, 15, 25],
                }),
                "window_size": 3,
                "real_data_ratio": 0.5,
                "generalize_ratio": 0.5,
                "interpolate_amount": "d"
            },
            "expected_shapes": {
                "train": (2, 3, 2),
                "test": (1, 3, 2),
                "generalize": (2, 3, 2)
            },
        },
    ]

    for test in test_cases:
        df = test["input"]["df"]
        window_size = test["input"]["window_size"]
        real_data_ratio = test["input"]["real_data_ratio"]
        generalize_ratio = test["input"]["generalize_ratio"]
        expected_shapes = test["expected_shapes"]
        interpolate_amount = test["input"]["interpolate_amount"]

        ans_train_np, ans_test_np, ans_generalize_np = harvest_data_with_interpolate(
            df, window_size, real_data_ratio, generalize_ratio, interpolate_amount)

        assert ans_train_np.shape == expected_shapes[
            "train"], f"Test {test['name']} failed: expected train shape {expected_shapes['train']}, got {ans_train_np.shape}"

        assert ans_test_np.shape == expected_shapes[
            "test"], f"Test {test['name']} failed: expected test shape {expected_shapes['test']}, got {ans_test_np.shape}"
        assert ans_generalize_np.shape == expected_shapes[
            "generalize"], f"Test {test['name']} failed: expected generalize shape {expected_shapes['generalize']}, got {ans_generalize_np.shape}"

        print(f"Test {test['name']} passed.")


def test_get_patients_from_dict_as_np():
    test_cases = [
        {
            "name": "Test 1 - Basic functionality",
            "input": {
                "patients_dict": {
                    "patient1": np.array([[1, 2], [3, 4]]),
                    "patient2": np.array([[5, 6]]),
                    "patient3": np.array([[7, 8], [9, 10]])
                },
                "black_list": ["patient2"],
                "in_or_not": False
            },
            "expected": np.array([[1, 2], [3, 4], [7, 8], [9, 10]])
        },
        {
            "name": "Test 2 - All patients in black_list",
            "input": {
                "patients_dict": {
                    "patient1": np.array([[1, 2]]),
                    "patient2": np.array([[3, 4]]),
                    "patient3": np.array([[5, 6]])
                },
                "black_list": ["patient1", "patient2", "patient3"],
                "in_or_not": True
            },
            "expected": np.array([[1, 2], [3, 4], [5, 6]])
        },
        {
            "name": "Test 3 - No patients in black_list",
            "input": {
                "patients_dict": {
                    "patient1": np.array([[1, 2]]),
                    "patient2": np.array([[3, 4]]),
                    "patient3": np.array([[5, 6]])
                },
                "black_list": ["patient4"],
                "in_or_not": False
            },
            "expected": np.array([[1, 2], [3, 4], [5, 6]])
        },
        {
            "name": "Test 4 - Empty black_list",
            "input": {
                "patients_dict": {
                    "patient1": np.array([[1, 2]]),
                    "patient2": np.array([[3, 4]]),
                    "patient3": np.array([[5, 6]])
                },
                "black_list": [],
                "in_or_not": True
            },
            "expected": np.array([])
        },
        {
            "name": "Test 5 - Empty patients_dict",
            "input": {
                "patients_dict": {},
                "black_list": ["patient1"],
                "in_or_not": True
            },
            "expected": np.array([])
        }
    ]

    for test in test_cases:
        patients_dict = test["input"]["patients_dict"]
        black_list = test["input"]["black_list"]
        in_or_not = test["input"]["in_or_not"]
        expected = test["expected"]

        try:
            result = get_patients_from_dict_as_np(patients_dict, black_list, in_or_not)
            assert np.array_equal(result, expected), f"Test {test['name']} failed: expected {expected}, got {result}"
            print(f"Test {test['name']} passed.")
        except AssertionError as e:
            raise e