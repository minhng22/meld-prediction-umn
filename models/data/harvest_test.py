import numpy as np
from models.data.harvest import find_train_test


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
        train, test_arr = find_train_test(test['arr'], test['window_size'], test['min_original_ratio'], test['arr_is_original'])
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