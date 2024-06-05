from torch.nn import LSTM, Linear, Conv1d, ReLU, MaxPool1d, Module


class CNNLSTMModel(Module):
    def __init__(self, num_layers, hidden_size, dropout_lstm, num_obs, num_pred, num_feature_input, num_feature_output):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.num_feature_ip = num_feature_input
        self.num_feature_op = num_feature_output

        # LSTM layers
        self.encoder_lstm = LSTM(
            self.num_feature_ip,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_lstm,
        )

        # Output layer
        self.fc = Linear(hidden_size, self.num_feature_op)

        self.encoder_cnn = Conv1d(
            in_channels=num_obs,
            out_channels=num_pred,
            kernel_size=1,
        )
        self.relu = ReLU()
        self.maxpool = MaxPool1d(kernel_size=1)

    def forward(self, x):
        cnn_op = self.encoder_cnn(x)
        relu_op = self.relu(cnn_op)
        maxpool_op = self.maxpool(relu_op)

        ec_op, (_, _) = self.encoder_lstm(maxpool_op)

        return self.fc(ec_op)