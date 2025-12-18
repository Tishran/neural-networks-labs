"""
Evaluation metrics for sequence-to-sequence models.

Implements:
- Character-level accuracy
- Sequence-level (exact match) accuracy
- Levenshtein distance
- Position-wise error analysis
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of edits (insert, delete, substitute) to transform s1 to s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of operations
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein distance (0 to 1).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Normalized distance (0 = identical, 1 = completely different)
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s1, s2) / max_len


def character_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute character-level accuracy.

    Args:
        predictions: List of predicted strings
        targets: List of target strings

    Returns:
        Ratio of correctly predicted characters
    """
    total_chars = 0
    correct_chars = 0

    for pred, tgt in zip(predictions, targets):
        max_len = max(len(pred), len(tgt))
        for i in range(max_len):
            total_chars += 1
            if i < len(pred) and i < len(tgt) and pred[i] == tgt[i]:
                correct_chars += 1

    return correct_chars / total_chars if total_chars > 0 else 0.0


def sequence_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute sequence-level (exact match) accuracy.

    Args:
        predictions: List of predicted strings
        targets: List of target strings

    Returns:
        Ratio of exactly matched sequences
    """
    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for pred, tgt in zip(predictions, targets) if pred == tgt)
    return correct / len(predictions)


def mean_levenshtein_distance(predictions: List[str], targets: List[str]) -> float:
    """
    Compute mean Levenshtein distance.

    Args:
        predictions: List of predicted strings
        targets: List of target strings

    Returns:
        Mean edit distance
    """
    if len(predictions) == 0:
        return 0.0

    distances = [levenshtein_distance(pred, tgt) for pred, tgt in zip(predictions, targets)]
    return np.mean(distances)


def calculate_metrics(
    predictions: List[str],
    targets: List[str]
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        predictions: List of predicted strings
        targets: List of target strings

    Returns:
        Dictionary with all metrics
    """
    return {
        'char_accuracy': character_accuracy(predictions, targets),
        'seq_accuracy': sequence_accuracy(predictions, targets),
        'mean_levenshtein': mean_levenshtein_distance(predictions, targets),
        'mean_normalized_levenshtein': np.mean([
            normalized_levenshtein(p, t) for p, t in zip(predictions, targets)
        ])
    }


def position_wise_accuracy(
    predictions: List[str],
    targets: List[str],
    max_position: Optional[int] = None
) -> Dict[int, Tuple[int, int]]:
    """
    Compute accuracy at each position in the sequence.

    Args:
        predictions: List of predicted strings
        targets: List of target strings
        max_position: Maximum position to consider

    Returns:
        Dictionary mapping position to (correct_count, total_count)
    """
    position_stats = defaultdict(lambda: [0, 0])  # [correct, total]

    for pred, tgt in zip(predictions, targets):
        max_len = max(len(pred), len(tgt))
        if max_position:
            max_len = min(max_len, max_position)

        for i in range(max_len):
            position_stats[i][1] += 1
            if i < len(pred) and i < len(tgt) and pred[i] == tgt[i]:
                position_stats[i][0] += 1

    return {k: tuple(v) for k, v in position_stats.items()}


def confusion_matrix_for_position(
    predictions: List[str],
    targets: List[str],
    position: int,
    vocab: List[str]
) -> np.ndarray:
    """
    Compute confusion matrix for a specific position.

    Args:
        predictions: List of predicted strings
        targets: List of target strings
        position: Position index to analyze
        vocab: List of vocabulary characters

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    vocab_to_idx = {c: i for i, c in enumerate(vocab)}
    num_classes = len(vocab) + 1  # +1 for "no character" (padding/truncation)
    no_char_idx = len(vocab)

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, tgt in zip(predictions, targets):
        pred_char = pred[position] if position < len(pred) else None
        tgt_char = tgt[position] if position < len(tgt) else None

        pred_idx = vocab_to_idx.get(pred_char, no_char_idx) if pred_char else no_char_idx
        tgt_idx = vocab_to_idx.get(tgt_char, no_char_idx) if tgt_char else no_char_idx

        matrix[tgt_idx, pred_idx] += 1

    return matrix


def analyze_errors_by_length(
    predictions: List[str],
    targets: List[str],
    decimals: List[int]
) -> Dict[int, Dict[str, float]]:
    """
    Analyze errors grouped by input number length.

    Args:
        predictions: List of predicted strings
        targets: List of target strings
        decimals: List of decimal numbers corresponding to each sample

    Returns:
        Dictionary mapping length to metrics for that length
    """
    length_groups = defaultdict(lambda: {'preds': [], 'tgts': []})

    for pred, tgt, num in zip(predictions, targets, decimals):
        length = len(str(num))
        length_groups[length]['preds'].append(pred)
        length_groups[length]['tgts'].append(tgt)

    results = {}
    for length, data in sorted(length_groups.items()):
        results[length] = calculate_metrics(data['preds'], data['tgts'])
        results[length]['count'] = len(data['preds'])

    return results


def analyze_errors_by_range(
    predictions: List[str],
    targets: List[str],
    decimals: List[int],
    ranges: List[Tuple[int, int]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze errors grouped by number ranges.

    Args:
        predictions: List of predicted strings
        targets: List of target strings
        decimals: List of decimal numbers
        ranges: List of (min, max) tuples for grouping

    Returns:
        Dictionary mapping range string to metrics
    """
    if ranges is None:
        ranges = [(1, 10), (11, 100), (101, 500), (501, 1000),
                  (1001, 2000), (2001, 3000), (3001, 3999)]

    range_groups = defaultdict(lambda: {'preds': [], 'tgts': []})

    for pred, tgt, num in zip(predictions, targets, decimals):
        for min_val, max_val in ranges:
            if min_val <= num <= max_val:
                key = f"{min_val}-{max_val}"
                range_groups[key]['preds'].append(pred)
                range_groups[key]['tgts'].append(tgt)
                break

    results = {}
    for key, data in range_groups.items():
        if data['preds']:
            results[key] = calculate_metrics(data['preds'], data['tgts'])
            results[key]['count'] = len(data['preds'])

    return results


class MetricsTracker:
    """Track metrics over training epochs."""

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, metrics: Dict[str, float], epoch: int):
        """Add metrics for an epoch."""
        for key, value in metrics.items():
            self.history[key].append((epoch, value))

    def get(self, metric_name: str) -> List[Tuple[int, float]]:
        """Get history for a specific metric."""
        return self.history[metric_name]

    def get_best(self, metric_name: str, mode: str = 'max') -> Tuple[int, float]:
        """Get best value and epoch for a metric."""
        history = self.history[metric_name]
        if not history:
            return -1, 0.0

        if mode == 'max':
            return max(history, key=lambda x: x[1])
        else:
            return min(history, key=lambda x: x[1])

    def to_dict(self) -> Dict[str, List[Tuple[int, float]]]:
        """Convert to regular dictionary."""
        return dict(self.history)
