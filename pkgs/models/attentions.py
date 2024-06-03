from math import log

import torch
from torch import Tensor
from torch.nn import Module, Transformer, Dropout, Linear


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
        self.pe = pe

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(Module):
    def __init__(self, n_head, num_encoder_layers, n_head_factor, dropout_pos_encoding, num_decoder_layers,
                 dropout_transformer, activation_fn, num_pred):
        super().__init__()

        d_model = n_head * n_head_factor
        self.num_feature_ip = 2
        self.num_feature_op = 1
        self.num_pred = num_pred

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
            d_model, dropout_pos_encoding, 1)

        self.linear = Linear(self.num_feature_ip, d_model)

        self.fc = Linear(d_model, self.num_feature_op)

    def forward(self, src):
        src = self.linear(src)
        src = self.pos_encoding(src)

        tgt = torch.narrow(src, 1, src.size(1) - 1, 1)
        final_op = None

        for _ in range(self.num_pred):
            op = self.transformer(src, tgt)
            tgt = op

            src = torch.narrow(src, 1, 1, src.size(1) - 1)
            src = torch.cat((src, tgt), 1)

            if final_op is None:
                final_op = tgt
            else:
                final_op = torch.cat((final_op, tgt), 1)

        return self.fc(final_op)