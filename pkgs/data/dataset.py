from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

from pkgs.commons import meld_scaler_path, time_stamp_scaler_path
from pkgs.data.commons import split_and_convert_to_3d
from pkgs.data.scaler import save_fitted_scaler


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

        self.train_ips_meld, self.train_targets_meld = split_and_convert_to_3d(
            self.meld_sc.fit_transform(trains[:, :, 0]), num_obs, num_pred)
        self.train_ips_time, _ = split_and_convert_to_3d(
            self.time_sc.fit_transform(trains[:, :, 1]), num_obs, num_pred)

        # This transform works the same as using a sub-scaler. Each feature scale independently.
        # Run this code to verify how sklearn transform works
        # def test_3():
        #     scaler = StandardScaler()
        #     scaler.fit(np.random.random((2, 10)))
        #
        #     b = np.random.random((2, 5))
        #
        #     # use subsetscaler
        #     subset_scaler = StandardScaler()
        #     subset_scaler.mean_ = scaler.mean_[:5]
        #     subset_scaler.var_ = scaler.var_[:5]
        #     subset_scaler.scale_ = scaler.scale_[:5]
        #
        #     b1 = subset_scaler.transform(b)
        #     print(f"b1 {b1}")
        #
        #     # use zero
        #     b2 = scaler.transform(np.concatenate((b, np.zeros((2, 5))), axis=1))[:, :5]
        #     print(f"b2 {b2}")
        #
        #     # use rand
        #     # notice how b2 stays the same
        #     b2 = scaler.transform(np.concatenate((b, np.random.random((2, 5))), axis=1))[:, :5]
        #     print(f"b2 {b2}")
        self.test_ips_meld, _ = split_and_convert_to_3d(self.meld_sc.transform(tests[:, :, 0]), num_obs, num_pred)
        self.generalize_ips_meld, _ = split_and_convert_to_3d(self.meld_sc.transform(generalizes[:, :, 0]), num_obs,
                                                              num_pred)

        self.test_ips_time, _ = split_and_convert_to_3d(self.time_sc.transform(tests[:, :, 1]), num_obs, num_pred)
        self.generalize_ips_time, _ = split_and_convert_to_3d(self.time_sc.transform(generalizes[:, :, 1]), num_obs,
                                                              num_pred)

        save_fitted_scaler(self.meld_sc, meld_scaler_path(num_obs, num_pred))
        save_fitted_scaler(self.time_sc, time_stamp_scaler_path(num_obs, num_pred))

    def setup_automated_tool(self, meld_scaler, time_scaler, ips_meld, ips_time, num_obs, num_pred):
        self.meld_sc, self.time_sc = meld_scaler, time_scaler

        self.test_ips_meld, _ = split_and_convert_to_3d(
            self.meld_sc.transform(
                np.concatenate(
                    (ips_meld, np.zeros((ips_meld.shape[0], num_pred))),
                    axis=1
                )
            ),
            num_obs, num_pred, False
        )
        self.test_ips_time, _ = split_and_convert_to_3d(
            self.time_sc.transform(
                np.concatenate(
                    (ips_time, np.zeros((ips_time.shape[0], num_pred))),
                    axis=1
                )
            ),
            num_obs, num_pred, False
        )

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
