"""
Neural network models for Seq2Seq with Attention.

Implements:
- Encoder: LSTM/GRU with optional bidirectional support
- Decoder: LSTM/GRU with Luong attention
- Seq2Seq: Full encoder-decoder model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from .attention import LuongAttention


class Encoder(nn.Module):
    """
    RNN Encoder for sequence-to-sequence model.

    Supports LSTM and GRU cells, with optional bidirectional processing.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = 'lstm',
        bidirectional: bool = False,
        dropout: float = 0.0,
        embedding_type: str = 'learned',
        padding_idx: int = 0
    ):
        """
        Initialize encoder.

        Args:
            vocab_size: Size of source vocabulary
            embed_size: Embedding dimension
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            cell_type: 'lstm' or 'gru'
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout probability (applied between layers)
            embedding_type: 'learned' or 'onehot'
            padding_idx: Index of padding token
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding_type = embedding_type

        # Embedding layer
        if embedding_type == 'onehot':
            self.embed_size = vocab_size
            self.embedding = None
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)

        # RNN layer
        rnn_input_size = self.embed_size
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                rnn_input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=bidirectional, dropout=rnn_dropout
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                rnn_input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=bidirectional, dropout=rnn_dropout
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        # Dropout for embedding
        self.dropout = nn.Dropout(dropout)

        # Linear layer to combine bidirectional outputs for decoder initialization
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
            if self.cell_type == 'lstm':
                self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Encode source sequence.

        Args:
            src: Source indices [batch_size, src_len]
            src_lengths: Actual lengths of source sequences [batch_size]

        Returns:
            outputs: Encoder outputs [batch_size, src_len, hidden_size * num_directions]
            hidden: Final hidden state(s), transformed for decoder
        """
        batch_size = src.size(0)

        # Embed input
        if self.embedding_type == 'onehot':
            embedded = F.one_hot(src, num_classes=self.vocab_size).float()
        else:
            embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # Pack padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Run through RNN
        if self.cell_type == 'lstm':
            outputs, (hidden, cell) = self.rnn(packed)
        else:
            outputs, hidden = self.rnn(packed)
            cell = None

        # Unpack outputs
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Transform hidden states for decoder
        if self.bidirectional:
            # hidden: [num_layers * 2, batch, hidden] -> [num_layers, batch, hidden * 2]
            # Then project to decoder size
            hidden = self._combine_bidirectional(hidden)
            hidden = torch.tanh(self.fc_hidden(hidden))
            if cell is not None:
                cell = self._combine_bidirectional(cell)
                cell = torch.tanh(self.fc_cell(cell))

        if self.cell_type == 'lstm':
            return outputs, (hidden, cell)
        else:
            return outputs, (hidden,)

    def _combine_bidirectional(self, hidden: torch.Tensor) -> torch.Tensor:
        """Combine forward and backward hidden states."""
        # hidden: [num_layers * 2, batch, hidden]
        # Reshape to [num_layers, 2, batch, hidden]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        # Concatenate forward and backward: [num_layers, batch, hidden * 2]
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        return hidden

    @property
    def output_size(self) -> int:
        """Return the output size of encoder (for decoder to use)."""
        return self.hidden_size * self.num_directions


class Decoder(nn.Module):
    """
    RNN Decoder with Luong Attention.

    Implements attention-based decoding with various RNN cell types.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        encoder_hidden_size: int,
        num_layers: int = 1,
        cell_type: str = 'lstm',
        attention_method: str = 'general',
        dropout: float = 0.0,
        embedding_type: str = 'learned',
        padding_idx: int = 0,
        tie_embeddings: bool = False
    ):
        """
        Initialize decoder.

        Args:
            vocab_size: Size of target vocabulary
            embed_size: Embedding dimension
            hidden_size: Hidden state dimension
            encoder_hidden_size: Output size of encoder
            num_layers: Number of RNN layers
            cell_type: 'lstm' or 'gru'
            attention_method: 'dot', 'general', or 'concat'
            dropout: Dropout probability
            embedding_type: 'learned' or 'onehot'
            padding_idx: Index of padding token
            tie_embeddings: Whether to tie embedding and output weights
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.embedding_type = embedding_type

        # Embedding layer
        if embedding_type == 'onehot':
            self.embed_size = vocab_size
            self.embedding = None
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)

        # Attention mechanism
        self.attention = LuongAttention(
            hidden_size, method=attention_method, encoder_hidden_size=encoder_hidden_size
        )

        # RNN layer - input is embedding + context from previous step
        rnn_input_size = self.embed_size
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                rnn_input_size, hidden_size, num_layers,
                batch_first=True, dropout=rnn_dropout
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                rnn_input_size, hidden_size, num_layers,
                batch_first=True, dropout=rnn_dropout
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        # Concat layer: combines RNN output and context
        self.concat = nn.Linear(hidden_size + encoder_hidden_size, hidden_size)

        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        # Tie weights if requested
        if tie_embeddings and embedding_type == 'learned' and embed_size == hidden_size:
            self.output_projection.weight = self.embedding.weight

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        hidden: Tuple[torch.Tensor, ...],
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Decode one step.

        Args:
            tgt: Target token indices [batch_size, 1] or [batch_size]
            hidden: Previous hidden state(s)
            encoder_outputs: Encoder outputs [batch_size, src_len, encoder_hidden]
            encoder_mask: Mask for encoder outputs [batch_size, src_len]

        Returns:
            output: Output logits [batch_size, vocab_size]
            hidden: New hidden state(s)
            attention_weights: Attention distribution [batch_size, src_len]
        """
        # Ensure tgt is 2D
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(1)

        # Embed input
        if self.embedding_type == 'onehot':
            embedded = F.one_hot(tgt, num_classes=self.vocab_size).float()
        else:
            embedded = self.embedding(tgt)
        embedded = self.dropout(embedded)  # [batch, 1, embed_size]

        # Run RNN step
        if self.cell_type == 'lstm':
            rnn_output, (h_n, c_n) = self.rnn(embedded, (hidden[0], hidden[1]))
            new_hidden = (h_n, c_n)
        else:
            rnn_output, h_n = self.rnn(embedded, hidden[0])
            new_hidden = (h_n,)

        rnn_output = rnn_output.squeeze(1)  # [batch, hidden]

        # Compute attention
        context, attention_weights = self.attention(
            rnn_output, encoder_outputs, encoder_mask
        )

        # Combine RNN output and context (Luong's concat)
        concat_input = torch.cat([rnn_output, context], dim=1)
        attentional_hidden = torch.tanh(self.concat(concat_input))
        attentional_hidden = self.dropout(attentional_hidden)

        # Project to vocabulary
        output = self.output_projection(attentional_hidden)

        return output, new_hidden, attention_weights


class Seq2Seq(nn.Module):
    """
    Complete Sequence-to-Sequence model with attention.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2
    ):
        """
        Initialize Seq2Seq model.

        Args:
            encoder: Encoder module
            decoder: Decoder module
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            pad_idx: Padding token index
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with teacher forcing.

        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Actual source lengths [batch_size]
            tgt: Target sequences [batch_size, tgt_len]
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: Output logits [batch_size, tgt_len - 1, vocab_size]
            attention_weights: List of attention weights for each step
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        device = src.device

        # Encode source
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # Create encoder mask - must match encoder output size
        max_src_len = encoder_outputs.size(1)
        encoder_mask = self.create_mask(src[:, :max_src_len])

        # Initialize outputs
        outputs = []
        attention_weights_all = []

        # First decoder input is SOS token
        decoder_input = tgt[:, 0]

        # Decode step by step (excluding last token which has no next token)
        for t in range(1, tgt_len):
            output, hidden, attention_weights = self.decoder(
                decoder_input, hidden, encoder_outputs, encoder_mask
            )
            outputs.append(output)
            attention_weights_all.append(attention_weights)

            # Decide next input: teacher forcing or model prediction
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = tgt[:, t]
            else:
                decoder_input = output.argmax(dim=1)

        # Stack outputs: [batch, tgt_len - 1, vocab_size]
        outputs = torch.stack(outputs, dim=1)

        return outputs, attention_weights_all

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create mask for padding positions."""
        return src != self.pad_idx

    def encode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        """Encode source sequence and return encoder outputs, hidden, and mask."""
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        max_src_len = encoder_outputs.size(1)
        encoder_mask = self.create_mask(src[:, :max_src_len])
        return encoder_outputs, hidden, encoder_mask

    def decode_step(
        self,
        decoder_input: torch.Tensor,
        hidden: Tuple[torch.Tensor, ...],
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        """Single decoding step."""
        return self.decoder(decoder_input, hidden, encoder_outputs, encoder_mask)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    embed_size: int = 64,
    hidden_size: int = 128,
    num_layers: int = 1,
    cell_type: str = 'lstm',
    bidirectional: bool = False,
    attention_method: str = 'general',
    dropout: float = 0.1,
    embedding_type: str = 'learned',
    pad_idx: int = 0,
    sos_idx: int = 1,
    eos_idx: int = 2
) -> Seq2Seq:
    """
    Factory function to create Seq2Seq model with given configuration.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        embed_size: Embedding dimension
        hidden_size: RNN hidden dimension
        num_layers: Number of RNN layers
        cell_type: 'lstm' or 'gru'
        bidirectional: Whether encoder is bidirectional
        attention_method: 'dot', 'general', or 'concat'
        dropout: Dropout probability
        embedding_type: 'learned' or 'onehot'
        pad_idx: Padding token index
        sos_idx: SOS token index
        eos_idx: EOS token index

    Returns:
        Seq2Seq model
    """
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell_type=cell_type,
        bidirectional=bidirectional,
        dropout=dropout,
        embedding_type=embedding_type,
        padding_idx=pad_idx
    )

    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        encoder_hidden_size=encoder.output_size,
        num_layers=num_layers,
        cell_type=cell_type,
        attention_method=attention_method,
        dropout=dropout,
        embedding_type=embedding_type,
        padding_idx=pad_idx
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx
    )

    return model
