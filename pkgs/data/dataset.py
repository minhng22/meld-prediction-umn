from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from pkgs.data.commons import split


class SlidingWindowDataset(Dataset):
    def __init__(self):
        self.meld_sc, self.time_sc = None, None
        self.train_meld_original, self.tests_original, self.generalizes_original = None, None, None

        self.train_ips_meld, self.train_targets_meld = None, None
        self.test_ips_meld, self.test_targets_meld = None, None
        self.train_ips_time, self.generalize_ips_meld = None, None
        self.test_ips_time, self.generalize_ips_time = None, None

    def setup_full(self, trains, tests, generalizes, num_obs, num_pred):
        self.meld_sc = MinMaxScaler((0, 1))
        self.time_sc = MinMaxScaler((-1, 1))

        tests = np.array(tests)
        generalizes = np.array(generalizes)

        self.train_meld_original = trains[:, :, 0]

        self.tests_original = tests
        self.generalizes_original = generalizes

        self.train_ips_meld, self.train_targets_meld = split(
            self.meld_sc.fit_transform(trains[:, :, 0]), num_obs, num_pred)
        self.train_ips_time, _ = split(
            self.time_sc.fit_transform(trains[:, :, 1]), num_obs, num_pred)

        self.test_ips_meld, _ = split(self.meld_sc.transform(tests[:, :, 0]), num_obs, num_pred)
        self.generalize_ips_meld, _ = split(self.meld_sc.transform(generalizes[:, :, 0]), num_obs, num_pred)

        self.test_ips_time, _ = split(self.time_sc.transform(tests[:, :, 1]), num_obs, num_pred)
        self.generalize_ips_time, _ = split(self.time_sc.transform(generalizes[:, :, 1]), num_obs, num_pred)

    def setup_test(self, tests, num_obs, num_pred, meld_sc, time_sc):
        self.meld_sc = meld_sc
        self.time_sc = time_sc
        self.tests_original = tests
        self.test_ips_meld = split(self.meld_sc.transform(tests[:, :, 0]), num_obs, num_pred)
        self.test_ips_time, _ = split(self.time_sc.transform(tests[:, :, 1]), num_obs, num_pred)

    def __getitem__(self, i):
        train = np.concatenate((self.train_ips_meld[i], self.train_ips_time[i]), axis=1)
        target = self.train_targets_meld[i]

        return torch.from_numpy(train), torch.from_numpy(target)

    def __len__(self):
        return len(self.train_ips_meld)

    def get_original_meld_train(self):
        return np.reshape(self.train_meld_original,
                          (self.train_meld_original.shape[0], self.train_meld_original.shape[1], 1))

    def get_original_meld_test(self):
        return self.tests_original[:, :, 0]

    def get_original_meld_generalize(self):
        return self.generalizes_original[:, :, 0]

    def get_test_ips(self):
        return np.concatenate((self.test_ips_meld, self.test_ips_time), axis=2)

    def get_train_ips(self):
        return np.concatenate((self.train_ips_meld, self.train_ips_time), axis=2)

    def get_target_ips(self):
        return self.train_targets_meld

    def get_test_ip_meld(self):
        return self.test_ips_meld

    def get_generalize_ips(self):
        return np.concatenate((self.generalize_ips_meld, self.generalize_ips_time), axis=2)

    def get_generalize_ip_meld(self):
        return self.generalize_ips_meld