"""
Attention mechanisms for Seq2Seq models.

Implements Luong attention (dot, general, concat variants).
Reference: Luong et al., 2015 - "Effective Approaches to Attention-based Neural Machine Translation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LuongAttention(nn.Module):
    """
    Luong Attention mechanism.

    Supports three scoring methods:
    - 'dot': score(h_t, h_s) = h_t^T * h_s
    - 'general': score(h_t, h_s) = h_t^T * W_a * h_s
    - 'concat': score(h_t, h_s) = v_a^T * tanh(W_a * [h_t; h_s])
    """

    def __init__(
        self,
        hidden_size: int,
        method: str = 'general',
        encoder_hidden_size: Optional[int] = None
    ):
        """
        Initialize Luong attention.

        Args:
            hidden_size: Decoder hidden size
            method: Attention method ('dot', 'general', 'concat')
            encoder_hidden_size: Encoder hidden size (defaults to hidden_size)
        """
        super().__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size or hidden_size

        if method == 'general':
            self.W_a = nn.Linear(self.encoder_hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W_a = nn.Linear(hidden_size + self.encoder_hidden_size, hidden_size, bias=False)
            self.v_a = nn.Linear(hidden_size, 1, bias=False)
        elif method != 'dot':
            raise ValueError(f"Unknown attention method: {method}")

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden: Current decoder hidden state [batch_size, hidden_size]
            encoder_outputs: Encoder outputs [batch_size, src_len, encoder_hidden_size]
            encoder_mask: Mask for padding [batch_size, src_len] (True = valid, False = pad)

        Returns:
            context: Context vector [batch_size, encoder_hidden_size]
            attention_weights: Attention distribution [batch_size, src_len]
        """
        batch_size, src_len, _ = encoder_outputs.shape

        # Compute attention scores
        if self.method == 'dot':
            # decoder_hidden: [batch, hidden] -> [batch, hidden, 1]
            # encoder_outputs: [batch, src_len, hidden]
            # scores: [batch, src_len]
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)

        elif self.method == 'general':
            # Transform encoder outputs: [batch, src_len, hidden]
            transformed = self.W_a(encoder_outputs)
            scores = torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)

        elif self.method == 'concat':
            # Expand decoder hidden to match encoder outputs
            decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
            # Concatenate: [batch, src_len, hidden + encoder_hidden]
            concat = torch.cat([decoder_expanded, encoder_outputs], dim=2)
            scores = self.v_a(torch.tanh(self.W_a(concat))).squeeze(2)

        # Apply mask (set padding positions to -inf before softmax)
        if encoder_mask is not None:
            scores = scores.masked_fill(~encoder_mask, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)

        # Compute context vector
        # attention_weights: [batch, src_len] -> [batch, 1, src_len]
        # encoder_outputs: [batch, src_len, encoder_hidden]
        # context: [batch, encoder_hidden]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for more complex attention patterns.
    Optional enhancement beyond basic Luong attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, hidden]
            key: [batch, seq_len, hidden]
            value: [batch, seq_len, hidden]
            mask: [batch, seq_len]

        Returns:
            output: [batch, hidden]
            attention_weights: [batch, num_heads, seq_len]
        """
        batch_size, seq_len, _ = key.shape

        # Linear projections
        Q = self.W_q(query).unsqueeze(1)  # [batch, 1, hidden]
        K = self.W_k(key)  # [batch, seq_len, hidden]
        V = self.W_v(value)  # [batch, seq_len, hidden]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, 1, seq_len]

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch, heads, 1, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, self.hidden_size)

        output = self.W_o(context)

        return output, attention_weights.squeeze(2)  # [batch, heads, seq_len]
