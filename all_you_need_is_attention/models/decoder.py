import torch.nn as nn

from all_you_need_is_attention.models.layers.multi_head_attention import MultiHeadAttention
from all_you_need_is_attention.models.layers.positional_encoding import PositionalEncoding
from all_you_need_is_attention.models.layers.positionwise_feedforward import PositionwiseFeedforward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = PositionwiseFeedforward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        attn_output = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        attn_output = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)

        return tgt


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_feedforward, trans_dropout) for _ in range(num_layers)]
        )
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, pos_dropout)

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.pos_encoding(tgt)

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)

        return tgt
