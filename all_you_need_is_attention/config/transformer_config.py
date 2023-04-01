class TransformerConfig:
    def __init__(self):
        self.vocab_size = 10000  # Size of vocabulary
        self.d_model = 512  # Model dimensionality
        self.nhead = 8  # Number of attention heads
        self.num_encoder_layers = 6  # Number of encoder layers
        self.num_decoder_layers = 6  # Number of decoder layers
        self.dim_feedforward = 2048  # Feedforward dimension
        self.max_seq_length = 512  # Maximum sequence length
        self.pos_dropout = 0.1  # Dropout rate for positional encoding
        self.trans_dropout = 0.1  # Dropout rate for transformer layers
