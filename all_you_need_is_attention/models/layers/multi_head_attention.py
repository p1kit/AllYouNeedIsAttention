import math

import torch
import torch.nn as nn
from torch import matmul

from all_you_need_is_attention.models.utils.attention_utils import scaled_dot_product_attention

"""
    Linear
+---------------------+-----+
query   | (batch_size, seq_len, d_model) |
+---------------------+-----+
        |
        v
+--------------------------------+------------------------+
|                            MultiHeadAttention               |
|     +------------------------+--------+--------+           |
|     | (batch_size, seq_len, nhead, d_k) | (batch_size, seq_len, nhead, d_k) | (batch_size, seq_len, nhead, d_k) |
|     +------------------------+--------+--------+           |
|                                          |                 |
|                                          v                 |
|                                    Scaled Dot-Product      |
|                                          |                 |
|                                          v                 |
|    +-------------------------+----------------------+     |
|    | (batch_size, seq_len, nhead, d_k) | (batch_size, seq_len, nhead, d_k) |   |
|    +-------------------------+----------------------+     |
|                                          |                 |
|                                          v                 |
+------------------------------------------+-----------------+
                           |
                           v
              Linear
                           |
                           v
    (batch_size, seq_len, d_model)
"""


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    Applies scaled dot-product attention to the input queries, keys, and values,
    with separate linear projections of `d_model` dimensions for `nhead` heads.

    Args:
        d_model (int): Number of expected input features.
        nhead (int): Number of heads in the multi-head attention module.

    Attributes:
        d_model (int): Number of expected input features.
        nhead (int): Number of heads in the multi-head attention module.
        d_k (int): Dimensionality of each head.
        linear_q (nn.Linear): Linear projection layer for queries.
        linear_k (nn.Linear): Linear projection layer for keys.
        linear_v (nn.Linear): Linear projection layer for values.
        linear_out (nn.Linear): Linear projection layer for output.
    """

    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        """
        Applies multi-head attention to the input queries, keys, and values.

        Args:
            query (torch.Tensor): Tensor of shape `(batch_size, query_len, d_model)`
                representing the input queries.
            key (torch.Tensor): Tensor of shape `(batch_size, key_len, d_model)`
                representing the input keys.
            value (torch.Tensor): Tensor of shape `(batch_size, value_len, d_model)`
                representing the input values.
            attn_mask (torch.Tensor, optional): Tensor of shape `(batch_size, nhead, query_len, key_len)`
                representing the attention mask. If specified, masked positions are
                filled with a large negative value. Defaults to None.

        Returns:
            torch.Tensor: Tensor of shape `(batch_size, query_len, d_model)` representing
            the output of the multi-head attention module.
        """

        batch_size = query.size(0)

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # Split the last dimension into `nhead` heads.
        query = query.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)

        attn_scores = matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = matmul(attn_probs, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.linear_out(attn_output)

        return attn_output
