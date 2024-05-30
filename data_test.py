from data import interpolate
import pandas as pd
from commons import INPUT_PATH

# with_manual = True to read the data from INPUT_PATH and interpolate it.
def test_interpolate(with_manual=False):
    # manual testing
    # save the interpolated data to verify the correctness of the interpolate function
    if with_manual:
        df = pd.read_csv(INPUT_PATH)
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
                "patient_id": [1, 1, 1, 2, 2],
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-04 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-03 00:00:00"
                ],
                "score": [10, 20, 30, 15, 25]
            },
            "inter_amount": "d",
            "expected_len": 7,
            "expected_len_original": 5,
        },
        {
            "name": "Weekly interpolation with no gaps",
            "data": {
                "patient_id": [1, 1, 1, 2, 2],
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00", "2023-01-15 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00"
                ],
                "score": [10, 20, 30, 15, 25]
            },
            "inter_amount": "W",
            "expected_len": 5,
            "expected_len_original": 5,
        },
        {
            "name": "Weekly interpolation with gaps",
            "data": {
                "patient_id": [1, 1, 1, 2, 2],
                "timestamp": [
                    "2023-01-01 00:00:00", "2023-01-08 00:00:00", "2024-01-27 00:00:00",
                    "2023-01-01 00:00:00", "2023-01-15 00:00:00"
                ],
                "score": [10, 20, 30, 15, 25]
            },
            "inter_amount": "W",
            "expected_len": 8, # limit the forward fill to 1
            "expected_len_original": 5, # record at 2024-01-27 should not get omitted even when the forward fill starts from 2023-01-01
        }
    ]

    for test in test_cases:
        df = pd.DataFrame(test["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        result = interpolate(df, test["inter_amount"])

        msg = f"Test {test['name']} failed: expected {test['expected_len']} records, got {len(result)}. Interpolated data: \n{result}"
        assert len(result) == test["expected_len"], msg

        msg = f"Test {test['name']} failed: expected {test['expected_len_original']} original records, got {result['is_original'].sum()}. Interpolated data: \n{result}"
        assert result["is_original"].sum() == test["expected_len_original"], msg
    
    print("All tests passed.")

if __name__ == "__main__":
    test_interpolate()