import math
import numpy as np

def bottom_patient_by_len_record(d: dict, bottom_ratio):
    keys = list(d.keys())
    keys.sort(key=lambda x: d[x].shape[0])

    print("keys", keys)

    return keys[: math.ceil(len(keys) * bottom_ratio)]


def generate_timestep_for_plot(x, y):
    """
    Generates data T of shape (X, y) such that T[:, i] looks like [i+1, i+1, ..., i+1].

    Parameters:
    X (int): Number of rows.
    y (int): Number of columns.

    Returns:
    np.array: The generated data of shape (X, y).
    """
    T = np.zeros((x, y))
    for i in range(y):
        T[:, i] = i + 1
    return T
