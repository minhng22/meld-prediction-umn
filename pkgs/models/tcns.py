import random

import torch
from torch import nn
from torch.nn import LSTM, Linear, Module


def calculate_output_length(length_in, kernel_size, stride, padding, dilation):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class TCNModel(Module):
    # This model is less accurate than CNNLSTMModel. Abandoned.
    def __init__( 
            self, cnn_dropout, tcn_num_layers, num_obs, num_pred, num_feature_input, num_feature_output
    ):
        super(TCNModel, self).__init__()

        self.cnn_channels = []
        for i in range(tcn_num_layers):
            if i == tcn_num_layers - 1:
                self.cnn_channels.append(num_pred)
            else:
                self.cnn_channels.append(num_pred * random.randint(2, 10))

        cnn_layers = []
        output_len = num_feature_input

        for i in range(len(self.cnn_channels)):
            in_channels = num_obs if i == 0 else self.cnn_channels[i - 1]
            out_channels = self.cnn_channels[i]

            # These gives best performance in cnnlstm model.
            padding, dilation, kernel_size, stride = 0, 1, 1, 1

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

            # Calculate output length of Conv1d
            output_len = calculate_output_length(
                output_len, kernel_size, stride, padding, dilation
            )

            # Calculate output length of MaxPool1d
            output_len = calculate_output_length(
                output_len, kernel_size, stride, 0, dilation
            )

        self.tcn = nn.Sequential(*cnn_layers)

        # Output layer
        self.fc = Linear(output_len, num_feature_output)
        self.output_len = output_len

    def forward(self, x: torch.Tensor):
        tcn_op = self.tcn(x)
        return self.fc(tcn_op)


class TCNLSTMModel(Module):
    # This model is less accurate than CNNLSTMModel. Abandoned.
    def __init__(
            self, num_layers, hidden_size, dropout_lstm, cnn_dropout, tcn_num_layers, num_obs, num_pred, num_feature_input, num_feature_output
    ):
        super(TCNLSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.cnn_channels = []
        for i in range(tcn_num_layers):
            if i == tcn_num_layers - 1:
                self.cnn_channels.append(num_pred)
            else:
                self.cnn_channels.append(num_pred * random.randint(2, 10))

        cnn_layers = []
        output_len = num_feature_input

        for i in range(len(self.cnn_channels)):
            in_channels = num_obs if i == 0 else self.cnn_channels[i - 1]
            out_channels = self.cnn_channels[i]

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

            # Calculate output length of Conv1d
            output_len = calculate_output_length(
                output_len, kernel_size, stride, padding, dilation
            )
            # Calculate output length of MaxPool1d
            output_len = calculate_output_length(
                output_len, kernel_size, stride, 0, dilation
            )

        self.tcn = nn.Sequential(*cnn_layers)

        # LSTM layers
        self.encoder_lstm = LSTM(
            input_size=self.cnn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm,
        )

        # Output layer
        self.fc = Linear(hidden_size, num_feature_output)

    def forward(self, x: torch.Tensor):
        tcn_op = self.tcn(x)
        print(tcn_op.shape)
        ec_op, (_, _) = self.encoder_lstm(tcn_op)

        return self.fc(ec_op)