import torch.nn as nn

from all_you_need_is_attention.models.layers.multi_head_attention import MultiHeadAttention
from all_you_need_is_attention.models.layers.positional_encoding import PositionalEncoding
from all_you_need_is_attention.models.layers.positionwise_feedforward import PositionwiseFeedforward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = PositionwiseFeedforward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            max_seq_length,
            pos_dropout,
            trans_dropout
    ):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_feedforward, trans_dropout) for _ in range(num_layers)]
        )
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, pos_dropout)

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, src, src_mask=None):

        src = self.embedding(src)
        src = self.pos_encoding(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
