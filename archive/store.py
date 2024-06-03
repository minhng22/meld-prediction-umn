import math
import os
import random
import time
from math import log
import copy

from xgboost import XGBRegressor

from preprocess import interpolate_with_sliding_window

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import sem, t, norm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from torch import Tensor
from torch.nn import (LSTM, Conv1d, Dropout, HuberLoss, Linear, MaxPool1d,
                      Module, MultiheadAttention, ReLU, Transformer, Embedding)
from torch.utils.data import DataLoader, Dataset

from utils import get_input_and_validate_data
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import joblib
import torch.nn.functional as F
import xgboost
from utils import NUM_OBS, NUM_PRED

import statsmodels.tsa.seasonal as smt

# Define the number of observed and predicted time steps

NUM_FEATURE = 1
NUM_FEATURE_INPUT = 2
NUM_FEATURE_OUTPUT = 1
NUM_FEATURE_FILLER = NUM_FEATURE_OUTPUT
TOTAL_REC = NUM_OBS + NUM_PRED

WORD_SIZE = 3000
EMBEDDING_DIM = 1

NUM_OF_RECORD_FACTOR = 3
BATCH_SIZE = 256
NUM_EPOCH = 100
DEVICE = torch.device("cpu")
N_TRIALS = 1
LEARNING_RATE = 0.1
GAMMA_RATE = 0.9
PATIENCE = 15
MODELS = ["xgboost", "rfr", "attention_lstm"]  # "linear"  "attention_lstm" "xgboost", "rfr", "linear"
INPUT_PATH = "./data/meld2_080923.csv"
PREPROCESSED_TRAIN_DATA_PATH = "./preprocessed_train.npy"
PREPROCESSED_TEST_DATA_PATH = "./preprocessed_test.npy"
PREPROCESSED_GEN_DATA_PATH = "./preprocessed_gen.npy"
TARGET_COLOR = "steelblue"
PREDICT_COLOR = "crimson"

# Filler model params
FILLER_MODEL_NAME = "cnn_lstm"
USE_FILLER_MODEL = False

EXP_SETUP_NAME = "."
PATH_POST_FIX = "obs_" + str(NUM_OBS) + "_pred_" + str(NUM_PRED)

CONFIDENCE_LEVEL = 0.95

SEASONAL_DEC_FIG_PATH = EXP_SETUP_NAME + "/data_analysis/" + PATH_POST_FIX
RMSE_DAYS_FIG_PATH = EXP_SETUP_NAME + "/rmse_by_days/" + PATH_POST_FIX
BOX_PLOT_FIG_PATH = EXP_SETUP_NAME + "/figs_box/" + PATH_POST_FIX
DETREND_ANALYSIS_FIG_PATH = EXP_SETUP_NAME + "/trend_analysis/" + PATH_POST_FIX
LINE_PLOT_FIG_PATH = EXP_SETUP_NAME + "/figs/" + PATH_POST_FIX
SCATTER_PLOT_FIG_PATH = EXP_SETUP_NAME + "/figs_scatter/" + PATH_POST_FIX
LINEAR_FIG_PATH = EXP_SETUP_NAME + "/figs_linear"
MODEL_SAVE_PATH = EXP_SETUP_NAME + "/trained_models/" + PATH_POST_FIX
MODEL_SAVE_PATH_LINEAR = EXP_SETUP_NAME + "/trained_models/linear"
MODEL_CP_SAVE_PATH = EXP_SETUP_NAME + "/models_cp/" + PATH_POST_FIX
PI_FIG_PATH = EXP_SETUP_NAME + "/figs_pi/" + PATH_POST_FIX


def get_optuna_params(trial: optuna.trial.Trial, model_name, filler=False):
    if model_name == "autocorrelation_lstm":
        num_layers = "num_layers" if not filler else "num_layers_filler"
        num_heads = "num_heads" if not filler else "num_heads_filler"
        hidden_size = "hidden_size" if not filler else "hidden_size_filler"
        dropout_lstm = "dropout_lstm" if not filler else "dropout_lstm_filler"
        dropout_attn = "dropout_attn" if not filler else "dropout_attn_filler"

        return {
            "num_layers": trial.suggest_int(num_layers, 1, 4),
            "num_heads": trial.suggest_categorical(num_heads, [2, 3, 4, 5, 6, 8]),
            "hidden_size": trial.suggest_int(hidden_size, 120, 480, step = 120),
            "dropout_lstm": trial.suggest_float(dropout_lstm, 0, 0.5),
            "dropout_attn": trial.suggest_float(dropout_attn, 0, 0.5),
        }

    if model_name == "attention_lstm":
        num_layers = "num_layers" if not filler else "num_layers_filler"
        num_heads = "num_heads" if not filler else "num_heads_filler"
        hidden_size = "hidden_size" if not filler else "hidden_size_filler"
        dropout_lstm = "dropout_lstm" if not filler else "dropout_lstm_filler"
        dropout_attn = "dropout_attn" if not filler else "dropout_attn_filler"

        return {
            "num_layers": trial.suggest_int(num_layers, 1, 6),
            "num_heads": trial.suggest_categorical(num_heads, [2, 3, 4, 5, 6, 8]),
            "hidden_size": trial.suggest_int(hidden_size, 120, 840, step = 30),
            "dropout_lstm": trial.suggest_float(dropout_lstm, 0, 0.5),
            "dropout_attn": trial.suggest_float(dropout_attn, 0, 0.5),
        }
    if model_name == "cnn_lstm":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_int("hidden_size", 120, 720, step = 60),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0, 0.5),
        }
    if model_name == "tcn_lstm":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_int("hidden_size", 120, 480, step = 120),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0, 0.5),
            "cnn_dropout": trial.suggest_float("cnn_dropout", 0, 0.5),
            "cnn_kernel_size": trial.suggest_int("cnn_kernel_size", 2, 7),
            "tcn_num_layers": trial.suggest_int("tcn_num_layers", 1, 10),
        }
    if model_name == "tcn":
        return {
            "cnn_dropout": trial.suggest_float("cnn_dropout", 0, 0.5),
            "tcn_num_layers": trial.suggest_int("tcn_num_layers", 1, 10),
        }
    if model_name == "lstm_seq_2_seq":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_int("hidden_size", 128, 640, step = 128),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0, 0.5),
        }
    if model_name == "lstm":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_int("hidden_size", 120, 840, step = 30),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0, 0.5),
        }
    if model_name == "transformer":
        return {
            "n_head": trial.suggest_int("n_head", 6, 8),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 4, 8),
            "n_head_factor": trial.suggest_int("n_head_factor", 50, 70, step = 5),
            "dropout_pos_encoding": trial.suggest_float("dropout_lstm", 0, 0.5),
            "num_decoder_layers": trial.suggest_int("num_decoder_layers", 4, 8),
            "dropout_transformer": trial.suggest_float("dropout_lstm", 0, 0.5),
            "activation_fn": trial.suggest_categorical("activation_fn", ["relu", "gelu"]),
        }
    if model_name == "transformer_filler":
        return {
            "n_head": trial.suggest_int("n_head", 6, 10),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 4, 8),
            "n_head_factor": trial.suggest_int("n_head_factor", 50, 90, step = 5),
            "dropout_pos_encoding": trial.suggest_float("dropout_lstm", 0, 0.5),
            "num_decoder_layers": trial.suggest_int("num_decoder_layers", 4, 8),
            "dropout_transformer": trial.suggest_float("dropout_lstm", 0, 0.5),
            "activation_fn": trial.suggest_categorical("activation_fn", ["relu", "gelu"]),
        }


def get_model(model_name, s_s, filler=False):
    if model_name == "autocorrelation_lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = AutocorelationLSTMModel(
            num_layers=s_s["num_layers"],
            num_heads=s_s["num_heads"],
            hidden_size=s_s["hidden_size"],
            dropout_lstm=s_s["dropout_lstm"],
            dropout_attn=s_s["dropout_attn"],
        )
    if model_name == "attention_lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = AttentionAutoencoderLSTMModel(
            num_layers=s_s["num_layers"],
            num_heads=s_s["num_heads"],
            hidden_size=s_s["hidden_size"],
            dropout_lstm=s_s["dropout_lstm"],
            dropout_attn=s_s["dropout_attn"],
            filler=filler
        )
    if model_name == "cnn_lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = CNNLSTMModel(
            num_layers=s_s["num_layers"],
            hidden_size=s_s["hidden_size"],
            dropout_lstm=s_s["dropout_lstm"],
            filler=filler
        )
    if model_name == "tcn_lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = TCNLSTMModel(
            num_layers=s_s["num_layers"],
            hidden_size=s_s["hidden_size"],
            dropout_lstm=s_s["dropout_lstm"],
            cnn_dropout=s_s["cnn_dropout"],
            tcn_num_layers=s_s["tcn_num_layers"],
        )
    if model_name == "tcn":
        m = TCNModel(
            cnn_dropout=s_s["cnn_dropout"],
            tcn_num_layers=s_s["tcn_num_layers"],
        )
    if model_name == "lstm_seq_2_seq":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = LSTMSeq2SeqModel(
            num_layers=s_s["num_layers"],
            hidden_size=s_s["hidden_size"],
            drop_out=s_s["dropout_lstm"],
        )
    if model_name == "lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = LSTMModel(
            num_layers=s_s["num_layers"],
            hidden_size=s_s["hidden_size"],
            drop_out=s_s["dropout_lstm"],
        )
    if model_name == "transformer":
        m = TransformerModel(
            n_head=s_s["n_head"],
            num_encoder_layers=s_s["num_encoder_layers"],
            n_head_factor=s_s["n_head_factor"],
            dropout_pos_encoding=s_s["dropout_pos_encoding"],
            num_decoder_layers=s_s["num_decoder_layers"],
            dropout_transformer=s_s["dropout_transformer"],
            activation_fn=s_s["activation_fn"],
        )

    m.to(DEVICE)
    return m


class LinearModel(Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = Linear(3, 1) # day,month,year to meld

    def forward(self, x):
        x = self.fc(x)
        return x


class AttentionAutoencoderLSTMModel(Module):
    def __init__(self, num_layers, num_heads, hidden_size, dropout_lstm, dropout_attn, filler):
        super(AttentionAutoencoderLSTMModel, self).__init__()
        self.hidden_size = self.find_hideen(hidden_size, num_heads)
        self.num_feature_ip = NUM_FEATURE_INPUT if not filler else NUM_FEATURE_FILLER

        # LSTM layers
        self.encoder_lstm = LSTM(
            self.num_feature_ip,
            self.hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_lstm,
        )
        self.decoder_lstm = LSTM(
            self.num_feature_ip,
            self.hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_lstm,
        )

        # Attention mechanism
        self.attn = MultiheadAttention(
            self.hidden_size, num_heads, dropout=dropout_attn)

        # Output layer
        self.fc = Linear(self.hidden_size, NUM_FEATURE_OUTPUT)

        self.num_feature_decoder_lstm = NUM_PRED if not filler else NUM_OBS

    def find_hideen(self, hidden, num_head):
        h = hidden
        while h % num_head != 0:
            h += 1
        return h

    def forward(self, x):
        _, (ec_h, ec_c) = self.encoder_lstm(x)

        ctx_vt, _ = self.attn(ec_h, ec_h, ec_c)

        dec_op, (_, _) = self.decoder_lstm(
            torch.zeros(x.size(0), self.num_feature_decoder_lstm, self.num_feature_ip).to(
                DEVICE), (ec_h, ctx_vt)
        )

        return self.fc(dec_op)


class CNNLSTMModel(Module):
    def __init__(self, num_layers, hidden_size, dropout_lstm, filler):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.num_feature_ip = NUM_FEATURE_INPUT if not filler else NUM_FEATURE_FILLER

        # LSTM layers
        self.encoder_lstm = LSTM(
            self.num_feature_ip,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_lstm,
        )

        # Output layer
        self.fc = Linear(hidden_size, NUM_FEATURE_OUTPUT)

        padding = 0
        dilation = 1
        kernel_size = 1
        stride = 1

        self.encoder_cnn = Conv1d(
            in_channels=NUM_OBS,
            out_channels=NUM_PRED if not filler else NUM_OBS,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )
        self.relu = ReLU()
        self.maxpool = MaxPool1d(
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )

    def forward(self, x):
        cnn_op = self.encoder_cnn(x)
        relu_op = self.relu(cnn_op)
        maxpool_op = self.maxpool(relu_op)

        ec_op, (_, _) = self.encoder_lstm(maxpool_op)

        return self.fc(ec_op)


class TCNModel(Module):
    # This model is less accurate than CNNLSTMModel. Abandoned.
    def __init__(
            self,
            cnn_dropout,
            tcn_num_layers,
    ):
        super(TCNModel, self).__init__()

        def calculate_output_length(length_in, kernel_size, stride, padding, dilation):
            return (
                           length_in + 2 * padding - dilation * (kernel_size - 1) - 1
                   ) // stride + 1

        cnn_channels = []
        for i in range(tcn_num_layers):
            if i == tcn_num_layers - 1:
                cnn_channels.append(NUM_PRED)
            else:
                cnn_channels.append(NUM_PRED * random.randint(2, 10))

        cnn_layers = []
        output_length = NUM_FEATURE

        for i in range(len(cnn_channels)):
            in_channels = NUM_OBS if i == 0 else cnn_channels[i - 1]
            out_channels = cnn_channels[i]

            # These gives best performance in cnnlstm model.
            padding = 0
            dilation = 1
            kernel_size = 1
            stride = 1

            cnn_layers += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    stride=stride,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(cnn_dropout),
                nn.MaxPool1d(kernel_size=kernel_size,
                             stride=stride, dilation=dilation),
            ]

            # Calc output length
            output_length = calculate_output_length(
                output_length, kernel_size, stride, padding, dilation
            )
            # Maxpooling output length
            output_length = calculate_output_length(
                output_length, kernel_size, stride, 0, dilation
            )

        self.tcn = nn.Sequential(*cnn_layers)

        # Output layer
        self.fc = Linear(output_length, NUM_FEATURE)

    def forward(self, x: torch.Tensor):
        tcn_op = self.tcn(x)

        return self.fc(tcn_op)


class TCNLSTMModel(Module):
    # This model is less accurate than CNNLSTMModel. Abandoned.
    def __init__(
            self,
            num_layers,
            hidden_size,
            dropout_lstm,
            cnn_dropout,
            tcn_num_layers,
    ):
        super(TCNLSTMModel, self).__init__()

        def calculate_output_length(length_in, kernel_size, stride, padding, dilation):
            return (
                           length_in + 2 * padding - dilation * (kernel_size - 1) - 1
                   ) // stride + 1

        self.hidden_size = hidden_size

        cnn_channels = []
        for i in range(tcn_num_layers):
            if i == tcn_num_layers - 1:
                cnn_channels.append(NUM_PRED)
            else:
                cnn_channels.append(NUM_PRED * random.randint(2, 10))

        cnn_layers = []
        output_length = NUM_FEATURE

        for i in range(len(cnn_channels)):
            in_channels = NUM_OBS if i == 0 else cnn_channels[i - 1]
            out_channels = cnn_channels[i]

            # These gives best performance in cnnlstm model.
            padding = 0
            dilation = 1
            kernel_size = 1
            stride = 1

            cnn_layers += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    stride=stride,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(cnn_dropout),
                nn.MaxPool1d(kernel_size=kernel_size,
                             stride=stride, dilation=dilation),
            ]

            # Calc output length
            output_length = calculate_output_length(
                output_length, kernel_size, stride, padding, dilation
            )
            # Maxpooling output length
            output_length = calculate_output_length(
                output_length, kernel_size, stride, 0, dilation
            )

        self.tcn = nn.Sequential(*cnn_layers)

        # LSTM layers
        self.encoder_lstm = LSTM(
            output_length,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_lstm,
        )

        # Output layer
        self.fc = Linear(hidden_size, NUM_FEATURE)

    def forward(self, x: torch.Tensor):
        tcn_op = self.tcn(x)

        ec_op, (_, _) = self.encoder_lstm(tcn_op)

        return self.fc(ec_op)


# scale data
class SlidingWindowDataset(Dataset):
    def __init__(self, trains, tests, generalizes):
        self.meld_sc = MinMaxScaler((0, 1))
        self.time_sc = MinMaxScaler((-1, 1))

        tests = np.array(tests)
        generalizes = np.array(generalizes)

        self.train_meld_original = trains[:, :, 0]

        self.tests_original = tests
        self.generalizes_original = generalizes

        self.train_ips_meld, self.train_targets_meld = get_input_and_validate_data(
            self.meld_sc.fit_transform(trains[:, :, 0]), NUM_OBS, NUM_PRED
        )
        self.train_ips_time, _ = get_input_and_validate_data(
            self.time_sc.fit_transform(trains[:, :, 1]), NUM_OBS, NUM_PRED
        )

        self.test_ips_meld, _ = get_input_and_validate_data(
            self.meld_sc.transform(tests[:, :, 0]), NUM_OBS, NUM_PRED
        )
        self.generalize_ips_meld, _ = get_input_and_validate_data(
            self.meld_sc.transform(generalizes[:, :, 0]), NUM_OBS, NUM_PRED
        )

        self.test_ips_time, _ = get_input_and_validate_data(
            self.time_sc.transform(tests[:, :, 1]), NUM_OBS, NUM_PRED
        )
        self.generalize_ips_time, _ = get_input_and_validate_data(
            self.time_sc.transform(generalizes[:, :, 1]), NUM_OBS, NUM_PRED
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


def fake_patient_id(df):
    i = 0
    for _, g in df.groupby("patient_id"):
        df.loc[g.index, "patient_id"] = "fake_patient_id_" + str(i)
        i += 1
    return df


# This prepares the data to export.
def export_data_prep(data_path):
    df = pd.read_csv(data_path)

    df = df.dropna()

    df = sort_timestamp(df)

    df = mean_day(df)

    df = fake_patient_id(df)

    df = interpolate(df, "1D")

    df.drop(["timestamp"], axis=1, inplace=True)

    df.to_csv('./data/interpolated.csv')


class LSTMModel(Module):
    def __init__(self, num_layers, hidden_size, drop_out) -> None:
        super(LSTMModel, self).__init__()
        self.encoder_lstm = LSTM(
            NUM_FEATURE_INPUT, hidden_size, num_layers, batch_first=True, dropout=drop_out
        )
        self.fc = Linear(hidden_size, NUM_FEATURE_OUTPUT)

    def forward(self, x):
        if NUM_PRED > NUM_OBS:
            x = torch.cat(
                (x, torch.zeros(x.size(0), NUM_PRED - NUM_OBS, NUM_FEATURE_INPUT)), dim=1)
        x, (_, _) = self.encoder_lstm(x)
        x = self.fc(x)

        if NUM_OBS > NUM_PRED:
            x = x[:, -NUM_PRED:, :]
        return x


class LSTMSeq2SeqModel(Module):
    def __init__(self, num_layers, hidden_size, drop_out):
        super(LSTMSeq2SeqModel, self).__init__()

        # LSTM layers
        self.encoder_lstm = LSTM(
            NUM_FEATURE, hidden_size, num_layers, batch_first=True, dropout=drop_out
        )
        self.decoder_lstm = LSTM(
            NUM_FEATURE, hidden_size, num_layers, batch_first=True, dropout=drop_out
        )

        # Output layer
        self.fc = Linear(hidden_size, NUM_FEATURE)

    def forward(self, x: torch.Tensor):
        _, (ec_h, ec_c) = self.encoder_lstm(x)

        dec_op, (_, _) = self.decoder_lstm(
            torch.zeros(x.size(0), NUM_PRED, NUM_FEATURE).to(
                DEVICE), (ec_h, ec_c)
        )

        return self.fc(dec_op)


class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(Module):
    def __init__(self, n_head, num_encoder_layers, n_head_factor, dropout_pos_encoding, num_decoder_layers,
                 dropout_transformer, activation_fn):
        super().__init__()

        d_model = n_head * n_head_factor

        self.transformer = Transformer(
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
            d_model=d_model,
            dropout=dropout_transformer,
            activation=activation_fn
        )
        self.pos_encoding = PositionalEncoding(
            d_model, dropout_pos_encoding, EMBEDDING_DIM)

        self.linear = Linear(NUM_FEATURE_INPUT, d_model)

        self.fc = Linear(d_model, NUM_FEATURE_OUTPUT)

    def forward(self, src):
        src = self.linear(src)
        src = self.pos_encoding(src)

        tgt = torch.narrow(src, 1, src.size(1) - 1, 1)
        final_op = None

        for _ in range(NUM_PRED):
            op = self.transformer(src, tgt)
            tgt = op

            src = torch.narrow(src, 1, 1, src.size(1) - 1)
            src = torch.cat((src, tgt), 1)

            if final_op is None:
                final_op = tgt
            else:
                final_op = torch.cat((final_op, tgt), 1)

        return self.fc(final_op)


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]]
                               for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                             1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + \
                        delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                            1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                             1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(
            0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(
                0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(
                0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


def get_sklearn_model(model_name: str):
    if model_name == "evr":
        return ExtraTreesRegressor()
    if model_name == "rfr":
        return RandomForestRegressor()


def get_sklearn_model_params(model_name: str):
    if model_name == "evr":
        return {
            'n_estimators': [100, 200, 300],
            'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
            'max_depth': [None, 5, 10],
            'max_features': ['sqrt', 'log2'],
        }
    if model_name == "rfr":
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }


def run_sklearn_model(dataset: SlidingWindowDataset, model_name: str):
    params = get_sklearn_model_params(model_name)
    model = get_sklearn_model(model_name)

    grid_search = HalvingGridSearchCV(model, params, cv=10)

    train_ips = np.reshape(dataset.get_train_ips(),
                           (-1, NUM_OBS * NUM_FEATURE_INPUT))
    train_targets = np.reshape(
        dataset.get_target_ips(), (-1, NUM_PRED * NUM_FEATURE_OUTPUT))

    if NUM_OBS == 1:
        train_ips = np.ravel(train_ips)
    if NUM_PRED == 1:
        train_targets = np.ravel(train_targets)

    print(f"train shape: {train_ips.shape} {train_targets.shape}")

    grid_search.fit(train_ips, train_targets)

    return grid_search.best_estimator_


def run_xgboost_model(dataset: SlidingWindowDataset, model_name: str):
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1]
    }
    model = XGBRegressor()

    model_path = MODEL_SAVE_PATH + "/" + model_name + ".json"
    if os.path.exists(model_path):
        model.load_model(model_path)
        return model

    grid_search = HalvingGridSearchCV(model, param_grid, cv=10)

    train_ips = np.reshape(dataset.get_train_ips(), (-1, NUM_OBS * NUM_FEATURE_INPUT))
    train_targets = np.reshape(dataset.get_target_ips(), (-1, NUM_PRED * NUM_FEATURE_OUTPUT))

    if NUM_OBS == 1:
        train_ips = np.ravel(train_ips)
    if NUM_PRED == 1:
        train_targets = np.ravel(train_targets)

    print(f"train shape: {train_ips.shape} {train_targets.shape}")

    grid_search.fit(train_ips, train_targets)

    return grid_search.best_estimator_


def analyze_timestep_rmse(ip, op, exp_name, model_name):
    print(f"Calculating rmse for {exp_name}")
    if ip.shape[1] != op.shape[1]:
        raise ValueError("ip and op must have same shape")
    rmses = calculate_rmse_of_time_step(ip, op)

    print(f"rmses: {rmses}")

    plt.plot(list(range(len(rmses))), rmses)

    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.title('RMSE by day')

    if not os.path.exists(RMSE_DAYS_FIG_PATH):
        os.makedirs(RMSE_DAYS_FIG_PATH)

    plt.savefig(RMSE_DAYS_FIG_PATH + "/" + exp_name + "_" + model_name, bbox_inches="tight")
    plt.clf()


def exp_sklearn_model(dataset: SlidingWindowDataset, model_name: str):
    s = time.time()
    model_path = MODEL_SAVE_PATH + "/" + model_name + ".pkl"

    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
    else:
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        best_model = run_sklearn_model(dataset, model_name)
        joblib.dump(best_model, model_path)

    print(f"best model: {best_model}")

    # evaluate on test set
    print("evaluating on test data")
    test_ips = np.reshape(dataset.get_test_ips(), (-1, NUM_OBS * NUM_FEATURE_INPUT))  # reshape data into 2D
    tests_ops = best_model.predict(test_ips)

    tests_ops_full = inverse_scale_ops(dataset.get_test_ip_meld(), tests_ops, dataset.meld_sc)
    tests_ops = tests_ops_full[:, NUM_OBS:]
    tests = dataset.get_original_meld_test()[:, NUM_OBS:]

    print(f"test shape: {tests.shape} {tests_ops.shape}")
    print(f"R-square is: {r2_score(tests, tests_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(tests, tests_ops, squared=False):.4f}")

    analyze_timestep_rmse(tests, tests_ops, "test", model_name)
    analyze_ci_and_pi(tests, tests_ops, "test", model_name)

    plot_name = f"{model_name} R-square :{round(r2_score(tests, tests_ops), 2)} RMSE {round(mean_squared_error(tests, tests_ops, squared=False), 2)}"

    plot_box(tests, tests_ops, plot_name, model_name, "test")
    plot_line(dataset.get_original_meld_test(),
              tests_ops_full, plot_name, model_name, "test")

    # evaluate on generalization set
    print("evaluating on generalize data")
    generalizes_ips = np.reshape(
        dataset.get_generalize_ips(), (-1, NUM_OBS * NUM_FEATURE_INPUT))  # reshape data into 2D
    generalizes_ops = best_model.predict(generalizes_ips)

    generalizes_ops_full = inverse_scale_ops(
        dataset.get_generalize_ip_meld(), generalizes_ops, dataset.meld_sc)
    generalizes_ops = generalizes_ops_full[:, NUM_OBS:]
    generalizes = dataset.get_original_meld_generalize()[:, NUM_OBS:]

    print(f"R-square is: {r2_score(generalizes, generalizes_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(generalizes, generalizes_ops, squared=False):.4f}"
    )

    analyze_timestep_rmse(generalizes, generalizes_ops, "generalize", model_name)
    analyze_ci_and_pi(generalizes, generalizes_ops, "generalize", model_name)

    plot_name = f"{model_name} R-square :{round(r2_score(generalizes, generalizes_ops), 2)} " \
                f"RMSE {round(mean_squared_error(generalizes, generalizes_ops, squared=False), 2)}"
    plot_box(generalizes, generalizes_ops, plot_name, model_name, "generalize")
    plot_line(dataset.get_original_meld_generalize(),
              generalizes_ops_full, plot_name, model_name, "generalize")

    print(f"total experiment time: {time.time() - s} seconds")


def generate_time(N):
    arr = np.arange(1, N + 1)
    print('generate_time ', arr)
    centered_arr = np.reshape(arr, (1, N))
    print('generate_time shape ', centered_arr.shape)
    return centered_arr


# input is 2-d array, represent shape of meld scores. return 2-d array of according time
def get_time(D):
    print('get_time D', D)
    N = D[1]
    T = np.tile(generate_time(N), (D[0], 1))
    print('get_time T', T.shape)
    return T


def run_linear_model(ip, op):
    print(f"run_linear_model shape {ip.shape} {op.shape}")

    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop for 5 trials
    num_trials = 5
    num_epochs = 100
    batch_size = 10
    best_loss = float('inf')
    best_model = None

    for trial in range(num_trials):
        model = LinearModel()

        for epoch in range(num_epochs):
            for i in range(0, len(op), batch_size):
                X_batch = torch.Tensor(ip[i:i + batch_size])
                y_batch = torch.Tensor(op[i:i + batch_size])

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Trial {trial + 1}, Epoch {epoch}, Loss: {loss.item()}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model

    return best_model


def exp_linear_model(df):
    s = time.time()

    df = df.dropna(subset=['timestamp', 'score'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract date, time, and year into numpy arrays
    day_arr = df['timestamp'].dt.day.values
    month_arr = df['timestamp'].dt.month.values
    year_arr = df['timestamp'].dt.year.values
    meld = df['score'].values
    print(f"ip shape {meld.shape}, {day_arr.shape}, {month_arr.shape}, {year_arr.shape}")

    T = np.concatenate((
        np.reshape(day_arr, (day_arr.shape[0], 1)),
        np.reshape(month_arr, (month_arr.shape[0], 1)),
        np.reshape(year_arr, (year_arr.shape[0], 1)),
    ), axis = 1)
    meld = np.reshape(meld, (meld.shape[0], 1))
    print(f"meld.shape T.shape {meld.shape} {T.shape}")
    print(f"meld_train nan {np.any(np.isnan(meld))} {np.any(np.isnan(T))}")

    break_i = int(meld.shape[0] * 0.8)
    meld_train, meld_test = meld[:break_i, :], meld[:break_i, :]
    time_train, time_test =  T[:break_i, :], T[:break_i, :]

    model_name = "linear"
    model_path = MODEL_SAVE_PATH_LINEAR + "/" + model_name + ".pt"

    if os.path.exists(model_path):
        print('best model exists')
        model = torch.load(model_path)
    else:
        model = run_linear_model(time_train, meld_train)
        torch.save(model, model_path)

    def plot(pred, real, plot_name, ext):
        plt.scatter(real, pred)
        plt.plot(real, real, c=PREDICT_COLOR)

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.title(plot_name)
        plt.xlabel("real MELD")
        plt.ylabel("predict MELD")

        if not os.path.exists(LINEAR_FIG_PATH):
            os.makedirs(LINEAR_FIG_PATH)

        plt.savefig(LINEAR_FIG_PATH + "/" + ext +
                    "_" + model_name, bbox_inches="tight")
        plt.clf()

    train_meld_pred = model(torch.Tensor(time_train)).detach().numpy()
    print(f"train_meld_pred {train_meld_pred.shape} {meld_train.shape}")
    print(f"meld_train nan {np.any(np.isnan(train_meld_pred))} {np.any(np.isnan(meld_train))}")
    plot_name_train = f"{model_name} R-square :{round(r2_score(train_meld_pred, meld_train), 3)} RMSE {round(mean_squared_error(train_meld_pred, meld_train, squared=False), 3)}"

    plot(train_meld_pred, meld_train, plot_name_train, "train")

    test_meld_pred = model(torch.Tensor(time_test)).detach().numpy()
    print(f"test_meld_pred {test_meld_pred.shape} {meld_test.shape}")
    p_name = f"{model_name} R-square :{round(r2_score(test_meld_pred, meld_test), 3)} RMSE {round(mean_squared_error(test_meld_pred, meld_test, squared=False), 3)}"

    plot(test_meld_pred, meld_test, p_name, "test")

def exp_xgboost_model(dataset: SlidingWindowDataset, model_name: str):
    s = time.time()

    best_model = run_xgboost_model(dataset, model_name)

    print(f"best model: {best_model}")

    # evaluate on test set
    print("evaluating on test data")
    test_ips = np.reshape(dataset.get_test_ips(), (-1, NUM_OBS * NUM_FEATURE_INPUT))  # reshape data into 2D
    tests_ops = best_model.predict(test_ips)

    tests_ops_full = inverse_scale_ops(dataset.get_test_ip_meld(), tests_ops, dataset.meld_sc)
    tests_ops = tests_ops_full[:, NUM_OBS:]
    tests = dataset.get_original_meld_test()[:, NUM_OBS:]

    print(f"test shape: {tests.shape} {tests_ops.shape}")
    print(f"R-square is: {r2_score(tests, tests_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(tests, tests_ops, squared=False):.4f}")

    analyze_timestep_rmse(tests, tests_ops, "test", model_name)
    analyze_ci_and_pi(tests, tests_ops, "test", model_name)

    plot_name = f"{model_name} R-square :{round(r2_score(tests, tests_ops), 2)} RMSE {round(mean_squared_error(tests, tests_ops, squared=False), 2)}"

    plot_box(tests, tests_ops, plot_name, model_name, "test")
    plot_line(dataset.get_original_meld_test(),
              tests_ops_full, plot_name, model_name, "test")

    # evaluate on generalization set
    print("evaluating on generalize data")
    generalizes_ips = np.reshape(
        dataset.get_generalize_ips(), (-1, NUM_OBS * NUM_FEATURE_INPUT))  # reshape data into 2D
    generalizes_ops = best_model.predict(generalizes_ips)

    generalizes_ops_full = inverse_scale_ops(
        dataset.get_generalize_ip_meld(), generalizes_ops, dataset.meld_sc)
    generalizes_ops = generalizes_ops_full[:, NUM_OBS:]
    generalizes = dataset.get_original_meld_generalize()[:, NUM_OBS:]

    print(f"R-square is: {r2_score(generalizes, generalizes_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(generalizes, generalizes_ops, squared=False):.4f}"
    )

    analyze_timestep_rmse(generalizes, generalizes_ops, "generalize", model_name)
    analyze_ci_and_pi(generalizes, generalizes_ops, "generalize", model_name)

    plot_name = f"{model_name} R-square :{round(r2_score(generalizes, generalizes_ops), 2)} " \
                f"RMSE {round(mean_squared_error(generalizes, generalizes_ops, squared=False), 2)}"
    plot_box(generalizes, generalizes_ops, plot_name, model_name, "generalize")
    plot_line(dataset.get_original_meld_generalize(),
              generalizes_ops_full, plot_name, model_name, "generalize")

    model_path = MODEL_SAVE_PATH + "/" + model_name + ".json"
    best_model.save_model(model_path)
    print(f"total experiment time: {time.time() - s} seconds")


def run_statmodels(dataset: SlidingWindowDataset, model_name: str):
    params = get_sklearn_model_params(model_name)
    model = get_sklearn_model(model_name)

    grid_search = HalvingGridSearchCV(model, params, cv=2)

    train_ips = np.reshape(dataset.get_train_ips(),
                           (-1, NUM_OBS * NUM_FEATURE_INPUT))
    train_targets = np.reshape(
        dataset.get_target_ips(), (-1, NUM_PRED * NUM_FEATURE_OUTPUT))

    if NUM_OBS == 1:
        train_ips = np.ravel(train_ips)
    if NUM_PRED == 1:
        train_targets = np.ravel(train_targets)

    print(f"train shape: {train_ips.shape} {train_targets.shape}")

    grid_search.fit(train_ips, train_targets)

    return grid_search.best_estimator_


def exp_stat_models(dataset: SlidingWindowDataset, model_name: str):
    s = time.time()
    model_path = MODEL_SAVE_PATH + "/" + model_name + ".pkl"

    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
    else:
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        best_model = run_statmodels(dataset, model_name)
        joblib.dump(best_model, model_path)

    print(f"best model: {best_model}")

    # evaluate on test set
    print("evaluating on test data")
    test_ips = np.reshape(dataset.get_test_ips(), (-1, NUM_OBS * NUM_FEATURE_INPUT))  # reshape data into 2D
    tests_ops = best_model.predict(test_ips)

    tests_ops_full = inverse_scale_ops(dataset.get_test_ip_meld(), tests_ops, dataset.meld_sc)
    tests_ops = tests_ops_full[:, NUM_OBS:]
    tests = dataset.get_original_meld_test()[:, NUM_OBS:]

    print(f"test shape: {tests.shape} {tests_ops.shape}")
    print(f"R-square is: {r2_score(tests, tests_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(tests, tests_ops, squared=False):.4f}")

    analyze_timestep_rmse(tests, tests_ops, "test", model_name)
    analyze_ci_and_pi(tests, tests_ops, "test", model_name)

    plot_name = f"{model_name} R-square :{round(r2_score(tests, tests_ops), 2)} RMSE {round(mean_squared_error(tests, tests_ops, squared=False), 2)}"

    plot_box(tests, tests_ops, plot_name, model_name, "test")
    plot_line(dataset.get_original_meld_test(),
              tests_ops_full, plot_name, model_name, "test")

    # evaluate on generalization set
    print("evaluating on generalize data")
    generalizes_ips = np.reshape(
        dataset.get_generalize_ips(), (-1, NUM_OBS * NUM_FEATURE_INPUT))  # reshape data into 2D
    generalizes_ops = best_model.predict(generalizes_ips)

    generalizes_ops_full = inverse_scale_ops(
        dataset.get_generalize_ip_meld(), generalizes_ops, dataset.meld_sc)
    generalizes_ops = generalizes_ops_full[:, NUM_OBS:]
    generalizes = dataset.get_original_meld_generalize()[:, NUM_OBS:]

    print(f"R-square is: {r2_score(generalizes, generalizes_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(generalizes, generalizes_ops, squared=False):.4f}"
    )

    analyze_timestep_rmse(generalizes, generalizes_ops, "generalize", model_name)
    analyze_ci_and_pi(generalizes, generalizes_ops, "generalize", model_name)

    plot_name = f"{model_name} R-square :{round(r2_score(generalizes, generalizes_ops), 2)} " \
                f"RMSE {round(mean_squared_error(generalizes, generalizes_ops, squared=False), 2)}"

    plot_box(generalizes, generalizes_ops, plot_name, model_name, "generalize")
    plot_line(dataset.get_original_meld_generalize(),
              generalizes_ops_full, plot_name, model_name, "generalize")

    print(f"total experiment time: {time.time() - s} seconds")


def run_arima_model(D, exp_name):
    def dickey_fuller_test(data):
        result = adfuller(data)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    train, test = np.split(D, [NUM_OBS], axis=1)
    dickey_fuller_test(train.reshape(-1))

    # _, ax = plt.subplots(1, 2, figsize=(10, 5))
    # plot_pacf(np.diff(train.reshape(-1)), ax=ax[0])
    # plot_acf(np.diff(train.reshape(-1)), ax=ax[1])

    # plt.show()

    if exp_name == "test":
        p, d, q = 3, 0, 5
    elif exp_name == "generalize":
        p, d, q = 1, 0, 1

    model = ARIMA(train.reshape(-1), order=(p, d, q))
    model_fit = model.fit()

    # Forecasting
    pred = model_fit.forecast(steps=test.shape[0] * NUM_PRED)
    pred = pred.reshape(-1, NUM_PRED)

    print(f"R square is {r2_score(test, pred):.4f}")
    print(f"RMSE is: {mean_squared_error(test, pred, squared=False):.4f}")

    D_op = np.concatenate((train, pred), axis=1)
    print(D.shape)
    print(D_op.shape)

    plot_line(test, pred, "arima", exp_name)
    plot_box(test, pred, "arima", exp_name)


def optuna_objective(
        trial: optuna.trial.Trial, train_dataloader: DataLoader, model_name, filler_model
):
    params = get_optuna_params(trial, model_name)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    model = get_model(model_name, params)

    # Define loss function and optimizer
    trial_loss = float("inf")
    criterion = HuberLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=GAMMA_RATE)

    # Early stopping
    best_val_loss = float("inf")
    patient_cnt = 0
    should_stop = False

    # Batch training
    # After a trial, save the model
    model_save_path = MODEL_CP_SAVE_PATH + f"/{model_name}_trial_{trial.number}.pt"

    # Train the model
    s = time.time()
    print("Training model")
    for epoch in range(NUM_EPOCH):
        epoc_s = time.time()
        total_loss = 0.0
        for train_ip, train_target in train_dataloader:
            train_ip, train_target = train_ip.float(), train_target.float()

            # Pass through filler model
            if filler_model != None:
                meld, timestamp = train_ip[:, :, 0], train_ip[:, :, 1]
                meld = torch.reshape(meld, (meld.shape[0], meld.shape[1], 1)).to(DEVICE)
                timestamp = torch.reshape(timestamp, (timestamp.shape[0], timestamp.shape[1], 1)).to(DEVICE)

                meld_filled = filler_model(meld)
                meld = mask_minus_one(meld, meld_filled)
                train_ip = torch.cat((meld, timestamp), dim=2)

            # Forward pass
            train_op = model(train_ip.to(DEVICE))

            # Compute loss
            loss = criterion(train_op.to(DEVICE), train_target.to(DEVICE))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= len(train_dataloader)

        if total_loss < best_val_loss:
            best_val_loss = total_loss
            patient_cnt = 0
            trial.set_user_attr(key="model", value=model)
        else:
            patient_cnt += 1
            if patient_cnt >= PATIENCE:
                should_stop = True

        trial_loss = min(trial_loss, total_loss)
        lr_scheduler.step()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCH}, Val Loss: {total_loss:.4f}"
            + f". Finished in {time.time() - epoc_s} seconds, or {int((time.time() - epoc_s) / 60)} minutes"
        )

        if should_stop:
            print("Early stopping!")
            break

    print(
        f"Time taken to train model this trial {time.time() - s} seconds or {round((time.time() - s) / 60)} minutes"
    )

    torch.save(model, model_save_path)

    return trial_loss


def rand_fill_minus1(melds):
    n = melds.numel()
    m = int(round(n * 0.8))
    # alternative: indices = torch.randperm(n)[:m]
    indices = np.random.choice(n, m, replace=False)
    melds = melds.contiguous()
    melds.flatten()[indices] = -1

    return melds


def optuna_objective_filler_model(
        trial: optuna.trial.Trial, D: DataLoader, model_name):
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # === Fill nan values with dl model ===
    filler_model_params = get_optuna_params(trial, model_name, True)
    filler_model = get_model(model_name, filler_model_params, True)
    filler_model.to(DEVICE)

    # Define loss function and optimizer
    trial_loss = float("inf")
    criterion = HuberLoss()
    optimizer = optim.Adam(
        filler_model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=GAMMA_RATE)

    # Early stopping
    best_val_loss = float("inf")
    patient_cnt = 0
    should_stop = False

    # Train the model
    s = time.time()
    print("Training filler model")
    for epoch in range(NUM_EPOCH):
        epoc_s = time.time()
        total_loss = 0.0
        for train_ip, train_target in D:
            train_ip, train_target = train_ip.float(), train_target.float()

            rand_filled_data = rand_fill_minus1(train_ip)
            rand_filled_data = rand_filled_data.to(DEVICE)

            meld_filler_op = filler_model(rand_filled_data)

            filler_loss = criterion(meld_filler_op.to(DEVICE), train_target.to(DEVICE))
            # === END Fill nan values with dl model ===

            # Backward and optimize
            optimizer.zero_grad()

            filler_loss.backward()

            optimizer.step()

            total_loss += filler_loss.item()

        total_loss /= len(D)

        if total_loss < best_val_loss:
            best_val_loss = total_loss
            patient_cnt = 0
            trial.set_user_attr(key="model", value=filler_model)
        else:
            patient_cnt += 1
            if patient_cnt >= PATIENCE:
                should_stop = True

        trial_loss = min(trial_loss, total_loss)
        lr_scheduler.step()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCH}, Val Loss: {total_loss:.4f}"
            + f". Finished in {time.time() - epoc_s} seconds, or {int((time.time() - epoc_s) / 60)} minutes"
        )

        if should_stop:
            print("Early stopping!")
            break

    print(
        f"Time taken to train model this trial {time.time() - s} seconds or {round((time.time() - s) / 60)} minutes"
    )
    return trial_loss


def filter_minus_one(data):
    t = None
    for i in range(len(data)):
        d = data[i]
        if not np.any(d == -1):
            if t is None:
                t = np.reshape(d, (1, d.shape[0], d.shape[1]))
            else:
                t = np.concatenate((t, np.reshape(d, (1, d.shape[0], d.shape[1]))), axis=0)
    return t


def mask_minus_one(arr1, arr2):
    # Find the indices where arr1 is equal to -1
    mask = (arr1 == -1)

    # Use boolean indexing to replace the corresponding elements in arr1 with arr2
    t = copy.deepcopy(arr1)
    t[mask] = arr2[mask]

    return t


class FillerDataset(Dataset):
    def __init__(self, dataset: SlidingWindowDataset):
        data = dataset.get_original_meld_train()
        print(f"Original data shape {data.shape}")

        sc = MinMaxScaler((0, 1))
        data = data[:, :NUM_OBS, :]

        print(f"after extracting observed meld {data.shape}")
        data = filter_minus_one(data)  # don't use filled meld record
        print(f"after filter minus 1 {data.shape}")

        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
        data = sc.fit_transform(data)
        self.target = np.reshape(data, (data.shape[0], NUM_OBS, 1))  # this is just MELD, so 1 feature

        print(f"target data shape {self.target.shape}")

        self.ips = rand_fill_minus1(torch.from_numpy(self.target))
        print(f"ips shape {self.ips.shape}")

    def __getitem__(self, i):
        return self.ips[i], torch.from_numpy(self.target[i])

    def __len__(self):
        return len(self.ips)


def ex_optuna(train_dataloader: DataLoader, model_name, use_filler_model):
    def create_filler_data_loader(original_data_loader: DataLoader):
        return DataLoader(
            FillerDataset(original_data_loader.dataset), batch_size=BATCH_SIZE, shuffle=True)

    def find_best_model(train_dataloader, model_name, for_filler_model, filler_model):
        study = optuna.create_study(direction="minimize")

        if for_filler_model:
            filler_data_loader = create_filler_data_loader(copy.deepcopy(train_dataloader))
            study.optimize(
                lambda trial: optuna_objective_filler_model(
                    trial=trial, D=filler_data_loader, model_name=model_name
                ),
                n_trials=N_TRIALS,
                gc_after_trial=True,
            )
        else:
            study.optimize(
                lambda trial: optuna_objective(
                    trial=trial, train_dataloader=train_dataloader, model_name=model_name, filler_model=filler_model
                ),
                n_trials=N_TRIALS,
                gc_after_trial=True,
            )

        # Print best hyperparameters and loss
        best_trial = study.best_trial

        print(f"Best hyperparameters: {study.best_params}")

        best_model = best_trial.user_attrs["model"]

        return best_model

    if use_filler_model:
        filler_model_path = MODEL_SAVE_PATH + "/" + "filler_model_" + FILLER_MODEL_NAME + ".pt"
        if os.path.exists(filler_model_path):
            filler_model = torch.load(filler_model_path)
        else:
            filler_model = find_best_model(train_dataloader, FILLER_MODEL_NAME, True, None)
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            torch.save(filler_model, filler_model_path)
        return find_best_model(train_dataloader, model_name, False, filler_model)
    return find_best_model(train_dataloader, model_name, False, None)


def inverse_scale_ops(ips, ops, sc):
    ops = np.reshape(ops, (-1, NUM_PRED))
    ips = np.reshape(ips, (-1, NUM_OBS))

    tests_ops = np.concatenate((ips, ops), axis=1)

    return sc.inverse_transform(tests_ops)


def plot_seasonal_dec(trend, seasonal, resid, data_name):
    _, ax = plt.subplots(nrows=3, ncols=1, figsize=(4, 9))
    ax[0].plot(trend)
    ax[0].set_title("Trend")

    ax[1].plot(seasonal)
    ax[1].set_title("Seasonal")

    ax[2].plot(resid)
    ax[2].set_title("Residual")

    plt.subplots_adjust(hspace=0.5)

    if not os.path.exists(SEASONAL_DEC_FIG_PATH):
        os.makedirs(SEASONAL_DEC_FIG_PATH)

    plt.savefig(SEASONAL_DEC_FIG_PATH + "/" + data_name, bbox_inches='tight')
    plt.clf()


def seasonal_dec(data, data_name="", need_plot=False):
    trends, seasonals, resids = None, None, None
    for i in range(data.shape[0]):
        d = data[i]
        sd = smt.seasonal_decompose(
            d, model='additive', period=int(TOTAL_REC / 2))

        trend = np.reshape(sd.trend, (1, sd.trend.shape[0]))
        seasonal = np.reshape(sd.seasonal, (1, sd.seasonal.shape[0]))
        resid = np.reshape(sd.resid, (1, sd.resid.shape[0]))

        if trends is None:
            trends = trend
            seasonals = seasonal
            resids = resid
        else:
            trends = np.concatenate((trends, trend), axis=0)
            seasonals = np.concatenate((seasonals, seasonal), axis=0)
            resids = np.concatenate((resids, resid), axis=0)

    print(f"shape of trends: {trends.shape} {resids.shape} {seasonals.shape}")

    trends = np.mean(trends, axis=0)
    seasonals = np.mean(seasonals, axis=0)
    resids = np.mean(resids, axis=0)

    if need_plot:
        plot_seasonal_dec(trends, seasonals, resids, data_name)

    return trends, seasonals, resids


def has_na(data):
    return np.isnan(np.sum(data))


def calculate_rmse_of_time_step(ip, op):
    rmses = []
    for i in range(ip.shape[1]):
        ip_i = ip[:, i]
        op_i = op[:, i]
        rmse = np.sqrt(mean_squared_error(ip_i, op_i))
        rmses.append(round(rmse, 3))
    return rmses


def detrend_analysis(data, need_plot=False):
    t, _, _ = seasonal_dec(data)
    print(f"t has na: {has_na(t)}")

    detrended_data = data - t

    if need_plot:
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(4, 9))
        ax[0].plot(np.average(data, axis=0))
        ax[0].set_title("Original")

        ax[1].plot(np.average(detrended_data, axis=0))
        ax[1].set_title("Detrended")

        plt.subplots_adjust(hspace=0.5)

        if not os.path.exists(DETREND_ANALYSIS_FIG_PATH):
            os.makedirs(DETREND_ANALYSIS_FIG_PATH)

        plt.savefig(DETREND_ANALYSIS_FIG_PATH + "/detrended", bbox_inches='tight')
        plt.clf()

    print(f"shape of detrended data: {detrended_data.shape} shape of data {data.shape}")
    print(f"nil check {has_na(detrended_data)}")

    return np.nan_to_num(detrended_data)


def run_exp():
    print(f"Device: {DEVICE}")

    print(f"pre-processing data, experimenting on obs {NUM_OBS} pred {NUM_PRED}")
    s = time.time()
    df = pd.read_csv(INPUT_PATH)

    exp_trains, exp_tests, exp_generalizes = None, None, None

    if not os.path.exists(PREPROCESSED_TRAIN_DATA_PATH):
        print("getting new data")
        exp_trains, exp_tests, exp_generalizes = interpolate_with_sliding_window(
            df, TOTAL_REC)
        np.save(PREPROCESSED_TRAIN_DATA_PATH, exp_trains)
        np.save(PREPROCESSED_TEST_DATA_PATH, exp_tests)
        np.save(PREPROCESSED_GEN_DATA_PATH, exp_generalizes)
    else:
        exp_trains = np.load(PREPROCESSED_TRAIN_DATA_PATH)
        exp_tests = np.load(PREPROCESSED_TEST_DATA_PATH)
        exp_generalizes = np.load(PREPROCESSED_GEN_DATA_PATH)

    print(f"preprocessing data takes {time.time() - s} seconds")

    if not os.path.exists(MODEL_CP_SAVE_PATH):
        os.makedirs(MODEL_CP_SAVE_PATH)

    for model_name in MODELS:
        best_model = None

        print("=====================================")
        print(f"exp on model {model_name}")

        dataset = SlidingWindowDataset(exp_trains, exp_tests, exp_generalizes)

        if model_name in ["evr", "rfr"]:
            exp_sklearn_model(dataset, model_name)
            continue

        if model_name == "xgboost":
            exp_xgboost_model(dataset, model_name)
            continue

        if model_name == "linear":
            exp_linear_model(df)
            continue

        print("finding best model")
        s = time.time()

        model_path = MODEL_SAVE_PATH + "/" + model_name + ".pt"
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(model_path)

        if os.path.exists(model_path):
            print('best model exists')
            best_model = torch.load(model_path)
        else:
            dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            best_model = ex_optuna(dl, model_name, USE_FILLER_MODEL)

            print(f"finding best model among trials of {model_name}")

            for i in range(N_TRIALS):
                cp_model_path = MODEL_CP_SAVE_PATH + f"/{model_name}_trial_{i}.pt"

                if not os.path.exists(cp_model_path):
                    raise Exception("no model here")

                print(f'study model from trial {i}')
                m = torch.load(cp_model_path)

                print("evaluating on test data")

                test_ips = torch.from_numpy(dataset.get_test_ips()).float()
                tests_ops = m(test_ips.to(DEVICE)).to("cpu").detach().numpy()
                # inverse meld
                tests_ops_full = inverse_scale_ops(test_ips[:, :, 0], tests_ops,
                                                   dataset.meld_sc)
                tests_ops = tests_ops_full[:, NUM_OBS:]
                tests = dataset.get_original_meld_test()[:, NUM_OBS:]

                print(f"test shape: {tests.shape} {tests_ops.shape}")
                curr_r2 = r2_score(tests, tests_ops)
                curr_rmse = mean_squared_error(tests, tests_ops, squared=False)

                print(f"R-square is: {curr_r2}")
                print(f"RMSE is: {curr_rmse}")

                if best_model == None:
                    best_model = m
                    continue

                best_model_ops = best_model(test_ips.to(DEVICE)).to("cpu").detach().numpy()
                best_model_ops_full = inverse_scale_ops(test_ips[:, :, 0], best_model_ops,
                                                        dataset.meld_sc)
                best_model_ops = best_model_ops_full[:, NUM_OBS:]
                print(f"check best model op shape {best_model_ops.shape}")
                best_rmse = mean_squared_error(tests, best_model_ops, squared=False)

                if curr_rmse > best_rmse:
                    best_model = m

        print(f"evaluate best model of {model_name}")
        print(f"model params: {best_model}")

        print("evaluating on test data")
        test_ips = torch.from_numpy(dataset.get_test_ips()).float()
        tests_ops = best_model(test_ips.to(DEVICE))

        # inverse meld
        tests_ops_full = inverse_scale_ops(test_ips[:, :, 0], tests_ops.to("cpu").detach().numpy(), dataset.meld_sc)
        tests_ops = tests_ops_full[:, NUM_OBS:]
        tests = dataset.get_original_meld_test()[:, NUM_OBS:]

        print(f"test shape: {tests.shape} {tests_ops.shape}")
        print(f"R-square is: {r2_score(tests, tests_ops):.4f}")
        print(
            f"RMSE is: {mean_squared_error(tests, tests_ops, squared=False):.4f}")

        analyze_timestep_rmse(tests, tests_ops, "test", model_name)
        analyze_ci_and_pi(tests, tests_ops, "test", model_name)

        plot_name = f"{model_name} R-square :{round(r2_score(tests, tests_ops), 2)} RMSE {round(mean_squared_error(tests, tests_ops, squared=False), 2)}"

        plot_box(tests, tests_ops, plot_name, model_name, "test")
        plot_line(dataset.get_original_meld_test(),
                  tests_ops_full, plot_name, model_name, "test")

        print("evaluating on generalize data")
        generalizes_ips = torch.from_numpy(
            dataset.get_generalize_ips()).float()
        generalizes_ops = best_model(generalizes_ips.to(DEVICE))

        # inverse meld
        generalizes_ops_full = inverse_scale_ops(
            generalizes_ips[:, :, 0], generalizes_ops.to("cpu").detach().numpy(), dataset.meld_sc)
        generalizes_ops = generalizes_ops_full[:, NUM_OBS:]
        generalizes = dataset.get_original_meld_generalize()[:, NUM_OBS:]

        print(f"R-square is: {r2_score(generalizes, generalizes_ops):.4f}")
        print(
            f"RMSE is {mean_squared_error(generalizes, generalizes_ops, squared=False):.4f}"
        )

        analyze_timestep_rmse(generalizes, generalizes_ops, "generalize", model_name)
        analyze_ci_and_pi(generalizes, generalizes_ops, "generalize", model_name)

        plot_name = f"{model_name} R-square :{round(r2_score(generalizes, generalizes_ops), 2)} " \
                    f"RMSE {round(mean_squared_error(generalizes, generalizes_ops, squared=False), 2)}"

        plot_box(generalizes, generalizes_ops, plot_name, model_name, "generalize")
        plot_line(dataset.get_original_meld_generalize(),
                  generalizes_ops_full, plot_name, model_name, "generalize")

        torch.save(best_model, model_path)

        print(
            f"Model experiment takes {time.time() - s} seconds or {int((time.time() - s) / 60)} minutes")
        print(f"Finished experiment on model {model_name}")
        print("=====================================================")


def plot_line(y_target, y, plot_name, model_name, ext=""):
    print("plot_line")
    y_avg = np.average(y, axis=0)
    y_target_avg = np.average(y_target, axis=0)

    tsf_y = np.arange(1, y_avg.shape[0] + 1)

    plt.plot(tsf_y, y_avg, color=PREDICT_COLOR, label="prediction")
    plt.plot(tsf_y, y_target_avg, color=TARGET_COLOR, label="target")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    stk = int(y_avg.shape[0] / 10) if int(y_avg.shape[0] / 10) > 0 else 1
    plt.xticks(np.arange(1, y_avg.shape[0] + 1, stk))
    plt.title(plot_name)

    if not os.path.exists(LINE_PLOT_FIG_PATH):
        os.makedirs(LINE_PLOT_FIG_PATH)

    plt.savefig(LINE_PLOT_FIG_PATH + "/" + ext +
                "_" + model_name, bbox_inches="tight")
    plt.clf()


def plot_scatter(y_target, y, plot_name, model_name, ext=""):
    print("plot_scatter")
    print(y_target.shape)
    print(y.shape)

    tsf_y = np.arange(1, y_target.shape[1] + 1)
    tsf_y = np.reshape(tsf_y, (1, tsf_y.shape[0]))
    tsf_y = np.tile(tsf_y, (y.shape[0], 1))

    y = y.flatten()
    y_target = y_target.flatten()
    print(y_target.shape)
    print(y.shape)

    plt.scatter(tsf_y, y, c=PREDICT_COLOR)
    plt.scatter(tsf_y, y_target, c=TARGET_COLOR)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    plt.title(plot_name)

    if not os.path.exists(SCATTER_PLOT_FIG_PATH):
        os.makedirs(SCATTER_PLOT_FIG_PATH)

    plt.savefig(SCATTER_PLOT_FIG_PATH + "/" + ext +
                "_" + model_name, bbox_inches="tight")
    plt.clf()


def plot_box(y_target, y, plot_name, model_name, ext=""):
    print("plot_box")

    def create_df(y, y_target):
        df = pd.DataFrame(columns=['score', 'data', 'day'])

        for i in range(y_target.shape[1]):
            for score in y_target[:, i]:
                df = pd.concat([df, pd.DataFrame([{'score': score, 'data': 'target', 'day': i + 1}])],
                               ignore_index=True)
            for score in y[:, i]:
                df = pd.concat([df, pd.DataFrame([{'score': score, 'data': 'prediction', 'day': i + 1}])],
                               ignore_index=True)
        return df

    sns.boxplot(data=create_df(y, y_target), x='day', y='score',
                hue='data', palette={'target': TARGET_COLOR, 'prediction': PREDICT_COLOR})

    plt.title(plot_name)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    if not os.path.exists(BOX_PLOT_FIG_PATH):
        os.makedirs(BOX_PLOT_FIG_PATH)

    plt.savefig(BOX_PLOT_FIG_PATH + "/" + ext +
                "_" + model_name, bbox_inches="tight")
    plt.clf()


def analyze_ci_and_pi(target, prediction, exp_name, model_name):
    def analyze_ci(target, pred, exp_name, model_name):
        data = calculate_rmse_of_time_step(target, pred)

        n = len(data)
        mean = np.mean(data)
        standard_error = sem(data)
        t_value = t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1)
        margin_of_error = t_value * standard_error

        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        print(
            f"Confidence interval {CONFIDENCE_LEVEL * 100}% for RMSE for {exp_name} {model_name}: {lower_bound:.2f} {upper_bound:.2f}")

    def analyze_pi(target, prediction, exp_name, model_name):
        residuals = target - prediction

        residual_std = np.std(residuals)

        z_score = norm.ppf((100 - CONFIDENCE_LEVEL) / 200)

        margin_of_error = z_score * residual_std

        lower_bound = np.min(prediction - margin_of_error, axis=0)
        upper_bound = np.max(prediction + margin_of_error, axis=0)

        print(f"Prediction interval {CONFIDENCE_LEVEL * 100}% for RMSE for {exp_name} {model_name}")

        tsf_y = np.arange(1, lower_bound.shape[0] + 1)

        plt.plot(tsf_y, lower_bound, color=PREDICT_COLOR, label="lower bound")
        plt.plot(tsf_y, upper_bound, color=TARGET_COLOR, label="upper bound")

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        stk = int(lower_bound.shape[0] / 10) if int(lower_bound.shape[0] / 10) > 0 else 1
        plt.xticks(np.arange(1, lower_bound.shape[0] + 1, stk))
        plt.title(model_name)

        if not os.path.exists(PI_FIG_PATH):
            os.makedirs(PI_FIG_PATH)

        plt.savefig(PI_FIG_PATH + "/" + exp_name +
                    "_" + model_name, bbox_inches="tight")
        plt.clf()

    analyze_ci(target, prediction, exp_name, model_name)
    analyze_pi(target, prediction, exp_name, model_name)


def analyze_model():
    model_name = "linear"
    model_path = MODEL_SAVE_PATH + "/" + model_name + ".pt"

    if os.path.exists(model_path):
        print('best model exists')
        model = torch.load(model_path)

    print(model.fc.weight)
    print(model.fc.bias)


if __name__ == "__main__":
    run_exp()
    # analyze_model()
