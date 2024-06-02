import math

def bottom_patient_by_len_record(d: dict, bottom_ratio):
    keys = list(d.keys())
    keys.sort(key=lambda x: d[x].shape[0])

    print("keys", keys)

    return keys[: math.ceil(len(keys) * bottom_ratio)]