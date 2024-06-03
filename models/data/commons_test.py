import numpy as np
from models.data.commons import bottom_patient_by_len_record, generate_timestep_for_plot

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