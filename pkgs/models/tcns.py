import random

import torch
from torch import nn
from torch.nn import LSTM, Linear, Module


class TCNModel(Module):
    # This model is less accurate than CNNLSTMModel. Abandoned.
    def __init__( 
            self, cnn_dropout, tcn_num_layers, num_obs, num_pred
    ):
        super(TCNModel, self).__init__()
        self.num_feature = 1

        def calculate_output_length(length_in, kernel_size, stride, padding, dilation):
            return (
                           length_in + 2 * padding - dilation * (kernel_size - 1) - 1
                   ) // stride + 1

        cnn_channels = []
        for i in range(tcn_num_layers):
            if i == tcn_num_layers - 1:
                cnn_channels.append(num_pred)
            else:
                cnn_channels.append(num_pred * random.randint(2, 10))

        cnn_layers = []
        output_length = self.num_feature

        for i in range(len(cnn_channels)):
            in_channels = num_obs if i == 0 else cnn_channels[i - 1]
            out_channels = cnn_channels[i]

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

            # Calc output length
            output_length = calculate_output_length(
                output_length, kernel_size, stride, padding, dilation
            )
            # Max pooling output length
            output_length = calculate_output_length(
                output_length, kernel_size, stride, 0, dilation
            )

        self.tcn = nn.Sequential(*cnn_layers)

        # Output layer
        self.fc = Linear(cnn_channels[-1], self.num_feature)

    def forward(self, x: torch.Tensor):
        tcn_op = self.tcn(x)
        return self.fc(tcn_op)


class TCNLSTMModel(Module):
    # This model is less accurate than CNNLSTMModel. Abandoned.
    def __init__(
            self, num_layers, hidden_size, dropout_lstm, cnn_dropout, tcn_num_layers, num_obs, num_pred
    ):
        super(TCNLSTMModel, self).__init__()
        self.num_feature = 1
        self.hidden_size = hidden_size

        def calculate_output_length(length_in, kernel_size_l, stride_l, padding_l, dilation_l):
            return (length_in + 2 * padding_l - dilation_l * (kernel_size_l - 1) - 1) // stride_l + 1

        cnn_channels = []
        for i in range(tcn_num_layers):
            if i == tcn_num_layers - 1:
                cnn_channels.append(num_pred)
            else:
                cnn_channels.append(num_pred * random.randint(2, 10))

        cnn_layers = []
        output_length = self.num_feature

        for i in range(len(cnn_channels)):
            in_channels = num_obs if i == 0 else cnn_channels[i - 1]
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
            # Max pooling output length
            output_length = calculate_output_length(
                output_length, kernel_size, stride, 0, dilation
            )

        self.tcn = nn.Sequential(*cnn_layers)

        # LSTM layers
        self.encoder_lstm = LSTM(
            input_size=cnn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm,
        )

        # Output layer
        self.fc = Linear(hidden_size, self.num_feature)

    def forward(self, x: torch.Tensor):
        tcn_op = self.tcn(x)
        ec_op, (_, _) = self.encoder_lstm(tcn_op)

        return self.fc(ec_op)