import torch


def create_padding_mask(seq):
    seq = seq.eq(0)
    seq = seq.unsqueeze(1).unsqueeze(2)
    return seq


def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.bool()
    return mask
