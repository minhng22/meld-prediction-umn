import torch
from torch.nn import Module
from torch.nn import LSTM, MultiheadAttention, Linear

class AttentionAutoencoderLSTMModel(Module):
    def __init__(self, num_layers, num_heads, hidden_size, dropout_lstm, dropout_attn, num_pred, device):
        super(AttentionAutoencoderLSTMModel, self).__init__()
        self.hidden_size = self.find_hideen(hidden_size, num_heads)
        self.num_feature_ip = 2 # MELD and timestamp

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
        self.fc = Linear(self.hidden_size, 1)

        self.num_feature_decoder_lstm = num_pred
        self.device = device

    def find_hideen(self, hidden, num_head):
        h = hidden
        while h % num_head != 0:
            h += 1
        return h

    def forward(self, x):
        _, (ec_h, ec_c) = self.encoder_lstm(x)

        ctx_vt, _ = self.attn(ec_h, ec_h, ec_c)

        dec_op, (_, _) = self.decoder_lstm(
            torch.zeros(x.size(0), self.num_feature_decoder_lstm, self.num_feature_ip).to(self.device), (ec_h, ctx_vt)
        )

        return self.fc(dec_op)


class LSTMModel(Module):
    def __init__(self, num_layers, hidden_size, drop_out, num_obs, num_pred) -> None:
        super(LSTMModel, self).__init__()
        self.num_feature_ip = 2
        self.num_feature_op = 1
        self.num_obs = num_obs
        self.num_pred = num_pred

        self.encoder_lstm = LSTM(
            self.num_feature_ip, hidden_size, num_layers, batch_first=True, dropout=drop_out
        )
        self.fc = Linear(hidden_size, self.num_feature_op)

    def forward(self, x):
        if self.num_pred > self.num_obs:
            x = torch.cat(
                (x, torch.zeros(x.size(0), self.num_pred - self.num_obs, self.num_feature_ip)), dim=1)
        x, (_, _) = self.encoder_lstm(x)
        x = self.fc(x)

        if self.num_obs > self.num_pred:
            x = x[:, -self.num_pred:, :]
        return x