import torch.nn as nn

from all_you_need_is_attention.models.decoder import Decoder
from all_you_need_is_attention.models.encoder import Encoder


class BaseTransformer(nn.Module):
    """
    Implementation of the Transformer architecture.

    Attributes:
        encoder (Encoder): Instance of the Encoder class.
        decoder (Decoder): Instance of the Decoder class.
        linear (nn.Linear): PyTorch linear layer for final output transformation.

    Methods:
        forward: Defines the forward pass of the BaseTransformer model.
        from_config: Instantiates a BaseTransformer model from a TransformerConfig object.
    """
    def __init__(
            self,
            vocab_size,
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            max_seq_length,
            pos_dropout,
            trans_dropout
    ):
        super(BaseTransformer, self).__init__()

        self.encoder = Encoder(
            vocab_size,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            max_seq_length,
            pos_dropout,
            trans_dropout
        )

        self.decoder = Decoder(
            vocab_size,
            d_model,
            nhead,
            num_decoder_layers,
            dim_feedforward,
            max_seq_length,
            pos_dropout,
            trans_dropout
        )

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Defines the forward pass of the BaseTransformer model.

        Args:
            src (torch.Tensor): Tensor containing source input sequence.
            tgt (torch.Tensor): Tensor containing target input sequence.
            src_mask (torch.Tensor): Tensor containing source sequence mask.
            tgt_mask (torch.Tensor): Tensor containing target sequence mask.
            memory_mask (torch.Tensor): Tensor containing memory sequence mask.

        Returns:
            output (torch.Tensor): Tensor containing the output of the final linear layer.
        """

        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        output = self.linear(output)
        return output

    @classmethod
    def from_config(cls, config):
        model = cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            max_seq_length=config.max_seq_length,
            pos_dropout=config.pos_dropout,
            trans_dropout=config.trans_dropout
        )
        return model
