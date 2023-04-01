import torch.nn as nn


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(PositionwiseFeedforward, self).__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
