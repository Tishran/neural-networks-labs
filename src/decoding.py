"""
Decoding strategies for sequence generation.

Implements:
- Greedy decoding
- Top-K sampling
- Top-P (nucleus) sampling
- Beam search
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .models import Seq2Seq


@dataclass
class DecodingResult:
    """Result of decoding a sequence."""
    tokens: List[int]
    score: float
    attention_weights: List[torch.Tensor]


def greedy_decode(
    model: 'Seq2Seq',
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int
) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
    """
    Greedy decoding: always select the most probable token.

    Args:
        model: Seq2Seq model
        src: Source sequences [batch_size, src_len]
        src_lengths: Source lengths [batch_size]
        max_len: Maximum decoding length
        sos_idx: Start token index
        eos_idx: End token index

    Returns:
        predictions: List of token indices for each sample
        attention_weights: List of attention weights for each sample
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    with torch.no_grad():
        # Encode
        encoder_outputs, hidden, encoder_mask = model.encode(src, src_lengths)

        # Initialize
        decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        all_predictions = [[] for _ in range(batch_size)]
        all_attention = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            # Decode one step
            output, hidden, attn_weights = model.decode_step(
                decoder_input, hidden, encoder_outputs, encoder_mask
            )

            # Get most probable token
            predicted = output.argmax(dim=1)

            # Store predictions and attention for unfinished sequences
            for i in range(batch_size):
                if not finished[i]:
                    all_predictions[i].append(predicted[i].item())
                    all_attention[i].append(attn_weights[i].cpu())

            # Check for EOS
            finished = finished | (predicted == eos_idx)
            if finished.all():
                break

            decoder_input = predicted

    return all_predictions, all_attention


def top_k_decode(
    model: 'Seq2Seq',
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    k: int = 5,
    temperature: float = 1.0
) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
    """
    Top-K sampling: sample from the top K most probable tokens.

    Args:
        model: Seq2Seq model
        src: Source sequences [batch_size, src_len]
        src_lengths: Source lengths [batch_size]
        max_len: Maximum decoding length
        sos_idx: Start token index
        eos_idx: End token index
        k: Number of top tokens to consider
        temperature: Sampling temperature

    Returns:
        predictions: List of token indices for each sample
        attention_weights: List of attention weights for each sample
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    with torch.no_grad():
        encoder_outputs, hidden, encoder_mask = model.encode(src, src_lengths)

        decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        all_predictions = [[] for _ in range(batch_size)]
        all_attention = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            output, hidden, attn_weights = model.decode_step(
                decoder_input, hidden, encoder_outputs, encoder_mask
            )

            # Apply temperature
            logits = output / temperature

            # Top-K filtering
            top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)

            # Sample from top-K
            sampled_idx = torch.multinomial(probs, 1).squeeze(-1)
            predicted = top_k_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            for i in range(batch_size):
                if not finished[i]:
                    all_predictions[i].append(predicted[i].item())
                    all_attention[i].append(attn_weights[i].cpu())

            finished = finished | (predicted == eos_idx)
            if finished.all():
                break

            decoder_input = predicted

    return all_predictions, all_attention


def top_p_decode(
    model: 'Seq2Seq',
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    p: float = 0.9,
    temperature: float = 1.0
) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
    """
    Top-P (nucleus) sampling: sample from smallest set of tokens with cumulative probability >= p.

    Args:
        model: Seq2Seq model
        src: Source sequences [batch_size, src_len]
        src_lengths: Source lengths [batch_size]
        max_len: Maximum decoding length
        sos_idx: Start token index
        eos_idx: End token index
        p: Cumulative probability threshold
        temperature: Sampling temperature

    Returns:
        predictions: List of token indices for each sample
        attention_weights: List of attention weights for each sample
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    with torch.no_grad():
        encoder_outputs, hidden, encoder_mask = model.encode(src, src_lengths)

        decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        all_predictions = [[] for _ in range(batch_size)]
        all_attention = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            output, hidden, attn_weights = model.decode_step(
                decoder_input, hidden, encoder_outputs, encoder_mask
            )

            # Apply temperature
            logits = output / temperature
            probs = F.softmax(logits, dim=-1)

            # Sort probabilities
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability > p
            sorted_indices_to_remove = cumulative_probs > p
            # Keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Zero out removed probabilities
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Sample
            sampled_idx = torch.multinomial(sorted_probs, 1).squeeze(-1)
            predicted = sorted_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

            for i in range(batch_size):
                if not finished[i]:
                    all_predictions[i].append(predicted[i].item())
                    all_attention[i].append(attn_weights[i].cpu())

            finished = finished | (predicted == eos_idx)
            if finished.all():
                break

            decoder_input = predicted

    return all_predictions, all_attention


@dataclass
class BeamHypothesis:
    """A single beam hypothesis."""
    tokens: List[int]
    score: float
    hidden: Tuple[torch.Tensor, ...]
    attention: List[torch.Tensor]


def beam_search(
    model: 'Seq2Seq',
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    beam_size: int = 5,
    length_penalty: float = 0.6
) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
    """
    Beam search decoding.

    Args:
        model: Seq2Seq model
        src: Source sequences [batch_size, src_len]
        src_lengths: Source lengths [batch_size]
        max_len: Maximum decoding length
        sos_idx: Start token index
        eos_idx: End token index
        beam_size: Number of beams
        length_penalty: Length penalty (alpha in (5 + len)^alpha / 6^alpha)

    Returns:
        predictions: List of best token indices for each sample
        attention_weights: List of attention weights for each sample
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    all_predictions = []
    all_attention = []

    with torch.no_grad():
        # Process each sample separately for beam search
        for b in range(batch_size):
            # Get single sample
            src_single = src[b:b+1]
            src_len_single = src_lengths[b:b+1]

            # Encode
            encoder_outputs, hidden, encoder_mask = model.encode(src_single, src_len_single)

            # Expand for beam search
            encoder_outputs = encoder_outputs.expand(beam_size, -1, -1)
            encoder_mask = encoder_mask.expand(beam_size, -1)

            if model.encoder.cell_type == 'lstm':
                hidden = (
                    hidden[0].expand(-1, beam_size, -1).contiguous(),
                    hidden[1].expand(-1, beam_size, -1).contiguous()
                )
            else:
                hidden = (hidden[0].expand(-1, beam_size, -1).contiguous(),)

            # Initialize beams
            beams = [BeamHypothesis(
                tokens=[],
                score=0.0,
                hidden=hidden,
                attention=[]
            )]

            completed = []

            for step in range(max_len):
                if not beams:
                    break

                # Collect all hypotheses for batched processing
                all_tokens = []
                all_hiddens_h = []
                all_hiddens_c = []

                for beam in beams:
                    token = beam.tokens[-1] if beam.tokens else sos_idx
                    all_tokens.append(token)

                    if model.encoder.cell_type == 'lstm':
                        all_hiddens_h.append(beam.hidden[0])
                        all_hiddens_c.append(beam.hidden[1])
                    else:
                        all_hiddens_h.append(beam.hidden[0])

                # Batch process
                decoder_input = torch.tensor(all_tokens, device=device)
                num_beams = len(beams)

                if model.encoder.cell_type == 'lstm':
                    batch_hidden = (
                        torch.cat(all_hiddens_h, dim=1),
                        torch.cat(all_hiddens_c, dim=1)
                    )
                else:
                    batch_hidden = (torch.cat(all_hiddens_h, dim=1),)

                batch_enc_out = encoder_outputs[:num_beams]
                batch_enc_mask = encoder_mask[:num_beams]

                output, new_hidden, attn_weights = model.decode_step(
                    decoder_input, batch_hidden, batch_enc_out, batch_enc_mask
                )

                log_probs = F.log_softmax(output, dim=-1)

                # Expand beams
                new_beams = []
                for i, beam in enumerate(beams):
                    # Get top-k candidates for this beam
                    top_log_probs, top_indices = log_probs[i].topk(beam_size)

                    for j in range(beam_size):
                        new_token = top_indices[j].item()
                        new_score = beam.score + top_log_probs[j].item()

                        # Extract hidden state for this beam
                        if model.encoder.cell_type == 'lstm':
                            new_h = (
                                new_hidden[0][:, i:i+1, :].clone(),
                                new_hidden[1][:, i:i+1, :].clone()
                            )
                        else:
                            new_h = (new_hidden[0][:, i:i+1, :].clone(),)

                        new_beam = BeamHypothesis(
                            tokens=beam.tokens + [new_token],
                            score=new_score,
                            hidden=new_h,
                            attention=beam.attention + [attn_weights[i].cpu()]
                        )

                        if new_token == eos_idx:
                            # Apply length penalty
                            length = len(new_beam.tokens)
                            lp = ((5 + length) ** length_penalty) / (6 ** length_penalty)
                            new_beam.score = new_beam.score / lp
                            completed.append(new_beam)
                        else:
                            new_beams.append(new_beam)

                # Keep top beams
                new_beams.sort(key=lambda x: x.score, reverse=True)
                beams = new_beams[:beam_size]

            # Add remaining beams to completed
            completed.extend(beams)

            # Get best hypothesis
            if completed:
                best = max(completed, key=lambda x: x.score)
                # Remove EOS if present
                tokens = [t for t in best.tokens if t != eos_idx]
                all_predictions.append(tokens)
                all_attention.append(best.attention)
            else:
                all_predictions.append([])
                all_attention.append([])

    return all_predictions, all_attention


def decode_with_strategy(
    model: 'Seq2Seq',
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int,
    strategy: str = 'greedy',
    **kwargs
) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
    """
    Decode using specified strategy.

    Args:
        model: Seq2Seq model
        src: Source sequences
        src_lengths: Source lengths
        max_len: Maximum decoding length
        strategy: Decoding strategy ('greedy', 'beam', 'top_k', 'top_p')
        **kwargs: Additional arguments for the strategy

    Returns:
        predictions: List of token indices for each sample
        attention_weights: List of attention weights for each sample
    """
    sos_idx = model.sos_idx
    eos_idx = model.eos_idx

    if strategy == 'greedy':
        return greedy_decode(model, src, src_lengths, max_len, sos_idx, eos_idx)
    elif strategy == 'beam':
        beam_size = kwargs.get('beam_size', 5)
        length_penalty = kwargs.get('length_penalty', 0.6)
        return beam_search(model, src, src_lengths, max_len, sos_idx, eos_idx,
                          beam_size, length_penalty)
    elif strategy == 'top_k':
        k = kwargs.get('k', 5)
        temperature = kwargs.get('temperature', 1.0)
        return top_k_decode(model, src, src_lengths, max_len, sos_idx, eos_idx,
                           k, temperature)
    elif strategy == 'top_p':
        p = kwargs.get('p', 0.9)
        temperature = kwargs.get('temperature', 1.0)
        return top_p_decode(model, src, src_lengths, max_len, sos_idx, eos_idx,
                           p, temperature)
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")
