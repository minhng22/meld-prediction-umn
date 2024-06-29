import ast
import sys
import numpy as np

from pkgs.data.dataset import SlidingWindowDataset

supporting_num_obs = [5]
supporting_num_pred = [1, 3, 5, 7]

def main():
    meld_na_scores, time_stamps, num_pred, err = process_input()
    if err is not None:
        return
    meld_na_scores = np.array(meld_na_scores)
    time_stamps = np.array(time_stamps)

    data = np.concatenate((
        np.reshape(meld_na_scores, (1, len(meld_na_scores), 1)),
        np.reshape(time_stamps, (1, len(time_stamps), 1))), axis=2)

    data = SlidingWindowDataset()
    data.setup_test(data, len(meld_na_scores), num_pred, None, None)


def process_input():
    meld_na_scores = ast.literal_eval(sys.argv[1])
    if (not isinstance(meld_na_scores, list)
            or not all(isinstance(score, (int, float)) for score in meld_na_scores)
            or len(meld_na_scores) not in supporting_num_obs):
        print("Please provide a valid list of numbers for MELDNa scores.")
        return None, None, None, ValueError

    time_stamps = ast.literal_eval(sys.argv[2])
    if not isinstance(time_stamps, list) or len(time_stamps) not in supporting_num_obs:
        print("Please provide a valid list of date times for time_stamps.")
        return None, None, None, ValueError

    num_pred = int(ast.literal_eval(sys.argv[3]))
    if num_pred not in supporting_num_pred:
        print("Please provide a valid number of predicting MELD.")
        return None, None, None, ValueError

    if len(time_stamps) != len(meld_na_scores):
        print("Number of time_stamps does not match number of meld_na_scores.")
        return None, None, None, ValueError

    return meld_na_scores, time_stamps, num_pred, None

if __name__ == '__main__':
    main()
