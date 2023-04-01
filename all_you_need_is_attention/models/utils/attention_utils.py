import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, attn_mask=None):
    """
    Applies the scaled dot-product attention mechanism on the given query, key, and value tensors.

    Args:
        query (torch.Tensor): A tensor of shape (batch_size, num_queries, d_k) representing the query vectors.
        key (torch.Tensor): A tensor of shape (batch_size, num_keys, d_k) representing the key vectors.
        value (torch.Tensor): A tensor of shape (batch_size, num_values, d_v) representing the value vectors.
        attn_mask (torch.Tensor, optional): An optional tensor of shape (batch_size, num_queries, num_keys)
            representing the attention mask.

    Returns:
        output (torch.Tensor): A tensor of shape (batch_size, num_queries, d_v) representing the output of the attention
            mechanism.
        attn_probs (torch.Tensor): A tensor of shape (batch_size, num_queries, num_keys) representing the attention
            probabilities.
    """

    d_k = query.size(-1)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, value)
    return output, attn_probs


def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
