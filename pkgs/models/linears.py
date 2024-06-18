import numpy as np
import torch
from torch.nn import Module, Linear


class LinearModel(Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = Linear(3, 1) # day,month,year to meld

    def forward(self, x):
        x = self.fc(x)
        return x


class TimeSeriesLinearModel(Module):
    def __init__(self, num_obs, num_pred, num_feature_ip, num_feature_op):
        super(TimeSeriesLinearModel, self).__init__()
        self.fc = Linear(num_obs * num_feature_ip, num_pred * num_feature_op)
        self.num_pred = num_pred
        self.num_feature_op = num_feature_op

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        x = self.fc(x)
        return torch.reshape(x, (x.shape[0], self.num_pred, self.num_feature_op))