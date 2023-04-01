import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for Transformer models.

    Args:
        d_model (int): The size of the model's embedding dimension.
        max_seq_length (int): The maximum length of the input sequence.
        dropout (float): The dropout probability.

    Attributes:
        pos_enc (Tensor): The positional encoding matrix of shape `(max_seq_length, d_model)`.

    Methods:
        forward(x): Applies the positional encoding to the input tensor x.

    The positional encoding matrix is computed during initialization and stored as a buffer.
    During the forward pass, the positional encoding is added to the input tensor and
    passed through a dropout layer.

    """
    def __init__(self, d_model, max_seq_length, dropout):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Initialize the positional encoding matrix
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_enc = torch.zeros(max_seq_length, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)

        # Register the positional encoding matrix as a buffer
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:x.size(0), :]
        return self.dropout(x)
