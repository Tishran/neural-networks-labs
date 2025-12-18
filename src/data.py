"""
Dataset generation and data loading for Decimal-to-Roman translation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import random
from collections import Counter


# Roman numeral conversion rules
ROMAN_NUMERALS = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
]

# Numbers that use subtractive notation (harder cases)
HARD_NUMBERS = [
    4, 9, 14, 19, 24, 29, 34, 39, 40, 44, 49,
    90, 94, 99, 140, 190, 400, 440, 490,
    900, 940, 990, 1400, 1900, 1940, 1990,
    444, 494, 499, 944, 949, 999, 1444, 1494, 1499,
    1944, 1949, 1999, 2444, 2494, 2499, 2944, 2949, 2999,
    3444, 3494, 3499, 3944, 3949, 3999,
    888, 1888, 2888, 3888,  # Long outputs
]


def decimal_to_roman(num: int) -> str:
    """
    Convert decimal number to Roman numeral string.

    Extended to support numbers beyond 3999 by repeating 'M' for thousands.
    For example: 5000 = MMMMM, 10000 = ten M's, etc.
    """
    if num <= 0:
        raise ValueError(f"Number must be positive, got {num}")

    result = []

    # Handle thousands (extended range - just repeat M)
    thousands = num // 1000
    result.append('M' * thousands)
    num = num % 1000

    # Handle remaining with standard Roman numerals
    for value, numeral in ROMAN_NUMERALS:
        if value >= 1000:
            continue  # Already handled thousands
        while num >= value:
            result.append(numeral)
            num -= value

    return ''.join(result)


def roman_to_decimal(roman: str) -> int:
    """Convert Roman numeral string to decimal number."""
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    prev_value = 0
    for char in reversed(roman.upper()):
        value = roman_values[char]
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value
    return result


class Vocabulary:
    """Vocabulary for encoding/decoding sequences."""

    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'

    def __init__(self, tokens: List[str]):
        """
        Initialize vocabulary with given tokens.

        Args:
            tokens: List of unique tokens (excluding special tokens)
        """
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.tokens = self.special_tokens + list(tokens)
        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for idx, token in enumerate(self.tokens)}

        self.pad_idx = self.token2idx[self.PAD_TOKEN]
        self.sos_idx = self.token2idx[self.SOS_TOKEN]
        self.eos_idx = self.token2idx[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, sequence: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode string sequence to list of indices."""
        indices = [self.token2idx[char] for char in sequence]
        if add_sos:
            indices = [self.sos_idx] + indices
        if add_eos:
            indices = indices + [self.eos_idx]
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Decode list of indices to string."""
        tokens = [self.idx2token[idx] for idx in indices]
        if remove_special:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return ''.join(tokens)


def create_vocabularies() -> Tuple[Vocabulary, Vocabulary]:
    """Create source (decimal) and target (Roman) vocabularies."""
    decimal_chars = list('0123456789')
    roman_chars = list('IVXLCDM')

    src_vocab = Vocabulary(decimal_chars)
    tgt_vocab = Vocabulary(roman_chars)

    return src_vocab, tgt_vocab


class DecimalRomanDataset(Dataset):
    """Dataset for Decimal-to-Roman translation."""

    def __init__(
        self,
        numbers: List[int],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None
    ):
        """
        Initialize dataset.

        Args:
            numbers: List of decimal numbers
            src_vocab: Source vocabulary (decimal)
            tgt_vocab: Target vocabulary (Roman)
            max_src_len: Maximum source sequence length (for padding)
            max_tgt_len: Maximum target sequence length (for padding)
        """
        self.numbers = numbers
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Create pairs
        self.pairs = []
        for num in numbers:
            decimal_str = str(num)
            roman_str = decimal_to_roman(num)
            self.pairs.append((decimal_str, roman_str))

        # Calculate max lengths
        self.max_src_len = max_src_len or max(len(p[0]) for p in self.pairs)
        self.max_tgt_len = max_tgt_len or max(len(p[1]) for p in self.pairs) + 2  # +2 for SOS, EOS

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        decimal_str, roman_str = self.pairs[idx]

        # Encode sequences
        src_indices = self.src_vocab.encode(decimal_str)
        tgt_indices = self.tgt_vocab.encode(roman_str, add_sos=True, add_eos=True)

        # Store lengths before padding
        src_len = len(src_indices)
        tgt_len = len(tgt_indices)

        # Pad sequences
        src_padded = src_indices + [self.src_vocab.pad_idx] * (self.max_src_len - len(src_indices))
        tgt_padded = tgt_indices + [self.tgt_vocab.pad_idx] * (self.max_tgt_len - len(tgt_indices))

        return {
            'src': torch.tensor(src_padded, dtype=torch.long),
            'tgt': torch.tensor(tgt_padded, dtype=torch.long),
            'src_len': torch.tensor(src_len, dtype=torch.long),
            'tgt_len': torch.tensor(tgt_len, dtype=torch.long),
            'decimal': self.numbers[idx],
            'decimal_str': decimal_str,
            'roman_str': roman_str
        }


def create_datasets(
    min_num: int = 1,
    max_num: int = 3999,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_src_len: Optional[int] = None,
    max_tgt_len: Optional[int] = None,
    sample_size: Optional[int] = None,
    stratified: str = 'none'
) -> Tuple[DecimalRomanDataset, DecimalRomanDataset, DecimalRomanDataset, Vocabulary, Vocabulary]:
    """
    Create train, validation, and test datasets.

    Args:
        min_num: Minimum number to include
        max_num: Maximum number to include (can be >3999 for extended range)
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility
        max_src_len: Maximum source sequence length
        max_tgt_len: Maximum target sequence length
        sample_size: If provided, sample this many numbers from the range (for large ranges)
        stratified: Stratification method:
            - 'none': No stratification (random sampling)
            - 'input': Stratify by input length (number of digits)
            - 'output': Stratify by output length (Roman numeral length)
            - 'diversity': Stratify by output character diversity (best for varied patterns)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    random.seed(seed)

    all_numbers_full = list(range(min_num, max_num + 1))

    if stratified == 'diversity' and sample_size is not None:
        # Stratify by OUTPUT CHARACTER DIVERSITY
        # Group numbers by the ratio of non-M characters (more diverse = better)
        # Buckets: 0-10%, 10-20%, ..., 90-100% non-M characters
        diversity_buckets = {i: [] for i in range(11)}  # 0-10 representing 0-100%

        for n in all_numbers_full:
            roman = decimal_to_roman(n)
            if len(roman) == 0:
                continue
            non_m_ratio = (len(roman) - roman.count('M')) / len(roman)
            bucket_idx = min(10, int(non_m_ratio * 10))
            diversity_buckets[bucket_idx].append(n)

        # Shuffle each bucket
        for nums in diversity_buckets.values():
            random.shuffle(nums)

        # Prioritize high-diversity buckets (more non-M chars)
        # Give more weight to buckets with higher diversity
        buckets = list(range(11))
        remaining_samples = sample_size
        bucket_samples = {b: 0 for b in buckets}

        # First pass: equal allocation
        remaining_buckets = sum(1 for b in buckets if len(diversity_buckets[b]) > 0)
        for b in buckets:
            if len(diversity_buckets[b]) == 0:
                continue
            target = remaining_samples // remaining_buckets
            actual = min(target, len(diversity_buckets[b]))
            bucket_samples[b] = actual
            remaining_samples -= actual
            remaining_buckets -= 1

        # Second pass: redistribute remaining
        while remaining_samples > 0:
            distributed = False
            for b in buckets:
                if bucket_samples[b] < len(diversity_buckets[b]) and remaining_samples > 0:
                    bucket_samples[b] += 1
                    remaining_samples -= 1
                    distributed = True
            if not distributed:
                break

        # Collect samples
        all_numbers = []
        for b in buckets:
            all_numbers.extend(diversity_buckets[b][:bucket_samples[b]])

        random.shuffle(all_numbers)

    elif stratified == 'output' and sample_size is not None:
        # Stratify by OUTPUT length (Roman numeral length) for better diversity
        # Group numbers by their Roman numeral length
        length_to_nums = {}
        for n in all_numbers_full:
            roman_len = len(decimal_to_roman(n))
            if roman_len not in length_to_nums:
                length_to_nums[roman_len] = []
            length_to_nums[roman_len].append(n)

        # Shuffle each bucket
        for nums in length_to_nums.values():
            random.shuffle(nums)

        # Calculate samples per bucket with redistribution
        buckets = sorted(length_to_nums.keys())
        remaining_samples = sample_size
        bucket_samples = {b: 0 for b in buckets}

        # First pass: equal allocation capped by bucket size
        remaining_buckets = len(buckets)
        for b in buckets:
            target = remaining_samples // remaining_buckets
            actual = min(target, len(length_to_nums[b]))
            bucket_samples[b] = actual
            remaining_samples -= actual
            remaining_buckets -= 1

        # Second pass: redistribute remaining to larger buckets
        while remaining_samples > 0:
            distributed = False
            for b in buckets:
                if bucket_samples[b] < len(length_to_nums[b]) and remaining_samples > 0:
                    bucket_samples[b] += 1
                    remaining_samples -= 1
                    distributed = True
            if not distributed:
                break

        # Collect samples
        all_numbers = []
        for b in buckets:
            all_numbers.extend(length_to_nums[b][:bucket_samples[b]])

        random.shuffle(all_numbers)

    elif stratified == 'input' and sample_size is not None:
        # Stratify by INPUT length (number of digits)
        buckets = []
        lower = min_num
        while lower <= max_num:
            upper = min(lower * 10 - 1, max_num)
            if lower < 10:
                upper = min(9, max_num)
            bucket_nums = list(range(lower, upper + 1))
            if bucket_nums:
                random.shuffle(bucket_nums)
                buckets.append(bucket_nums)
            lower = upper + 1

        # Calculate samples per bucket with redistribution
        remaining_samples = sample_size
        remaining_buckets = len(buckets)
        bucket_sample_counts = [0] * len(buckets)

        for i, bucket in enumerate(buckets):
            target = remaining_samples // remaining_buckets
            actual = min(target, len(bucket))
            bucket_sample_counts[i] = actual
            remaining_samples -= actual
            remaining_buckets -= 1

        while remaining_samples > 0:
            distributed = False
            for i, bucket in enumerate(buckets):
                if bucket_sample_counts[i] < len(bucket) and remaining_samples > 0:
                    bucket_sample_counts[i] += 1
                    remaining_samples -= 1
                    distributed = True
            if not distributed:
                break

        all_numbers = []
        for i, bucket in enumerate(buckets):
            all_numbers.extend(bucket[:bucket_sample_counts[i]])

        random.shuffle(all_numbers)
    else:
        # No stratification - random sampling
        all_numbers = all_numbers_full
        random.shuffle(all_numbers)
        if sample_size is not None and sample_size < len(all_numbers):
            all_numbers = all_numbers[:sample_size]

    # Split
    n_total = len(all_numbers)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_numbers = all_numbers[:n_train]
    val_numbers = all_numbers[n_train:n_train + n_val]
    test_numbers = all_numbers[n_train + n_val:]

    # Create vocabularies
    src_vocab, tgt_vocab = create_vocabularies()

    # Determine max lengths from actual sampled data
    if max_src_len is None:
        max_src_len = max(len(str(n)) for n in all_numbers)

    if max_tgt_len is None:
        # Calculate longest Roman numeral from actual sampled numbers
        longest_len = max(len(decimal_to_roman(n)) for n in all_numbers)
        max_tgt_len = longest_len + 2  # +2 for SOS, EOS

    # Create datasets
    train_dataset = DecimalRomanDataset(train_numbers, src_vocab, tgt_vocab, max_src_len, max_tgt_len)
    val_dataset = DecimalRomanDataset(val_numbers, src_vocab, tgt_vocab, max_src_len, max_tgt_len)
    test_dataset = DecimalRomanDataset(test_numbers, src_vocab, tgt_vocab, max_src_len, max_tgt_len)

    return train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab


def create_dataloaders(
    train_dataset: DecimalRomanDataset,
    val_dataset: DecimalRomanDataset,
    test_dataset: DecimalRomanDataset,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test datasets."""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def get_dataset_statistics(dataset: DecimalRomanDataset) -> Dict:
    """
    Compute statistics about the dataset.

    Returns dict with:
        - src_lengths: distribution of source lengths
        - tgt_lengths: distribution of target lengths
        - digit_freq: frequency of each digit in source
        - roman_freq: frequency of each Roman numeral in target
        - number_range: (min, max) of numbers
    """
    src_lengths = []
    tgt_lengths = []
    digit_counts = Counter()
    roman_counts = Counter()

    for decimal_str, roman_str in dataset.pairs:
        src_lengths.append(len(decimal_str))
        tgt_lengths.append(len(roman_str))
        digit_counts.update(decimal_str)
        roman_counts.update(roman_str)

    return {
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths,
        'digit_freq': dict(digit_counts),
        'roman_freq': dict(roman_counts),
        'number_range': (min(dataset.numbers), max(dataset.numbers)),
        'num_samples': len(dataset)
    }
