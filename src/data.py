"""
Dataset generation and data loading for Decimal-to-Roman translation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import random
import re
from collections import Counter


# Roman numeral conversion rules (standard form for 1-3999)
ROMAN_NUMERALS = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
]

# Regex pattern for validating extended Roman numerals (n > 3999)
# Format: optional vinculum part (underscore-prefixed chars) + optional standard part
EXTENDED_ROMAN_PATTERN = re.compile(r'^(?:_[IVXLCDM])+[IVXLCDM]*$|^[IVXLCDM]+$')


def _standard_roman(num: int) -> str:
    """Convert number 1-3999 to standard Roman numeral."""
    if num <= 0 or num > 3999:
        raise ValueError(f"Standard Roman requires 1-3999, got {num}")
    result = []
    for value, numeral in ROMAN_NUMERALS:
        while num >= value:
            result.append(numeral)
            num -= value
    return ''.join(result)


def decimal_to_roman(num: int) -> str:
    """
    Convert decimal number to Roman numeral string.

    For 1-3999: Standard Roman numerals (I, V, X, L, C, D, M with subtractives).
    For > 3999: Vinculum notation using underscore prefix for x1000 multiplier.
                e.g., 4000 = _I_V, 50000 = _L, 100000 = _C

    Args:
        num: Positive integer to convert

    Returns:
        Roman numeral string
    """
    if num <= 0:
        raise ValueError(f"Number must be positive, got {num}")

    if num <= 3999:
        return _standard_roman(num)

    # Extended range: use vinculum (underscore prefix = x1000)
    thousands = num // 1000  # This part gets vinculum notation
    remainder = num % 1000   # This part is standard

    result = []

    # Convert thousands part to vinculum notation
    if thousands > 0:
        if thousands > 3999:
            raise ValueError(f"Number too large: {num}. Max supported is 3,999,999")
        thousands_roman = _standard_roman(thousands)
        # Prefix each character with underscore
        vinculum_part = ''.join(f'_{c}' for c in thousands_roman)
        result.append(vinculum_part)

    # Convert remainder to standard notation
    if remainder > 0:
        result.append(_standard_roman(remainder))

    return ''.join(result)


def roman_to_decimal(roman: str) -> int:
    """
    Convert Roman numeral string to decimal number.

    Supports both standard (1-3999) and extended vinculum notation (> 3999).

    Args:
        roman: Roman numeral string (may include underscore-prefixed chars)

    Returns:
        Decimal integer value
    """
    if not roman:
        raise ValueError("Empty Roman numeral string")

    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

    result = 0
    i = 0

    # Parse vinculum part (underscore-prefixed characters = x1000)
    vinculum_chars = []
    while i < len(roman) and roman[i] == '_':
        if i + 1 >= len(roman):
            raise ValueError(f"Invalid Roman numeral: underscore at end: {roman}")
        vinculum_chars.append(roman[i + 1])
        i += 2

    # Convert vinculum part using standard Roman parsing
    if vinculum_chars:
        vinculum_str = ''.join(vinculum_chars)
        prev_value = 0
        for char in reversed(vinculum_str):
            if char not in roman_values:
                raise ValueError(f"Invalid Roman character: {char}")
            value = roman_values[char]
            if value < prev_value:
                result -= value * 1000
            else:
                result += value * 1000
            prev_value = value

    # Parse standard part (remaining characters)
    standard_part = roman[i:]
    if standard_part:
        prev_value = 0
        for char in reversed(standard_part):
            if char not in roman_values:
                raise ValueError(f"Invalid Roman character: {char}")
            value = roman_values[char]
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value

    return result


def validate_roman(roman: str, num: int) -> bool:
    """
    Validate that a Roman numeral string is correctly formatted.

    For n > 3999: Must match pattern ^(?:_[IVXLCDM])+[IVXLCDM]*$
    For n <= 3999: Must match pattern ^[IVXLCDM]+$

    Also performs round-trip validation.
    """
    if num > 3999:
        if not EXTENDED_ROMAN_PATTERN.match(roman):
            return False
        # Must start with underscore for n > 3999
        if not roman.startswith('_'):
            return False
    else:
        if not re.match(r'^[IVXLCDM]+$', roman):
            return False

    # Round-trip check
    try:
        return roman_to_decimal(roman) == num
    except ValueError:
        return False


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


def create_vocabularies(extended: bool = False) -> Tuple[Vocabulary, Vocabulary]:
    """
    Create source (decimal) and target (Roman) vocabularies.

    Args:
        extended: If True, include '_' in target vocab for vinculum notation
    """
    decimal_chars = list('0123456789')
    roman_chars = list('IVXLCDM')
    if extended:
        roman_chars = ['_'] + roman_chars  # Add underscore for vinculum

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
    sample_size: Optional[int] = None
) -> Tuple[DecimalRomanDataset, DecimalRomanDataset, DecimalRomanDataset, Vocabulary, Vocabulary]:
    """
    Create train, validation, and test datasets.

    Args:
        min_num: Minimum number to include
        max_num: Maximum number to include (up to 3,999,999 with vinculum notation)
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility
        max_src_len: Maximum source sequence length
        max_tgt_len: Maximum target sequence length
        sample_size: If provided, sample this many numbers from the range

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    random.seed(seed)

    # Generate all numbers
    all_numbers = list(range(min_num, max_num + 1))
    random.shuffle(all_numbers)

    # Sample if requested
    if sample_size is not None and sample_size < len(all_numbers):
        all_numbers = all_numbers[:sample_size]

    # Split
    n_total = len(all_numbers)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_numbers = all_numbers[:n_train]
    val_numbers = all_numbers[n_train:n_train + n_val]
    test_numbers = all_numbers[n_train + n_val:]

    # Create vocabularies (extended if max_num > 3999)
    extended = max_num > 3999
    src_vocab, tgt_vocab = create_vocabularies(extended=extended)

    # Determine max lengths from actual sampled data
    if max_src_len is None:
        max_src_len = max(len(str(n)) for n in all_numbers)

    if max_tgt_len is None:
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


def run_tests():
    """Run sanity tests for Roman numeral conversion."""
    print("Running Roman numeral conversion tests...")

    # Test standard range (1-3999)
    standard_tests = [
        (1, 'I'), (4, 'IV'), (9, 'IX'), (49, 'XLIX'),
        (99, 'XCIX'), (499, 'CDXCIX'), (999, 'CMXCIX'),
        (3888, 'MMMDCCCLXXXVIII'), (3999, 'MMMCMXCIX')
    ]
    for num, expected in standard_tests:
        result = decimal_to_roman(num)
        assert result == expected, f"Failed: {num} -> {result}, expected {expected}"
        assert roman_to_decimal(result) == num, f"Round-trip failed for {num}"
        assert validate_roman(result, num), f"Validation failed for {num}"
    print(f"  [OK] Standard range (1-3999): {len(standard_tests)} tests passed")

    # Test extended range (> 3999)
    extended_tests = [
        (4000, '_I_V'),
        (5000, '_V'),
        (6000, '_V_I'),
        (10000, '_X'),
        (49999, '_X_L_I_XCMXCIX'),
        (50000, '_L'),
        (100000, '_C'),
        (500000, '_D'),
        (1000000, '_M'),
        (3999999, '_M_M_M_C_M_X_C_I_XCMXCIX'),
    ]
    for num, expected in extended_tests:
        result = decimal_to_roman(num)
        assert result == expected, f"Failed: {num} -> {result}, expected {expected}"
        assert roman_to_decimal(result) == num, f"Round-trip failed for {num}"
        assert validate_roman(result, num), f"Validation failed for {num}"
    print(f"  [OK] Extended range (> 3999): {len(extended_tests)} tests passed")

    # Test regex pattern for extended numbers
    for num in [4000, 49999, 50000, 100000]:
        roman = decimal_to_roman(num)
        assert EXTENDED_ROMAN_PATTERN.match(roman), f"Pattern mismatch for {num}: {roman}"
    print("  [OK] Regex pattern validation passed")

    print("All tests passed!")


if __name__ == '__main__':
    run_tests()

    # Demo with sample generation
    print("\n" + "=" * 60)
    print("Sample conversions (max_n=100000, samples=200)")
    print("=" * 60)

    train_ds, val_ds, test_ds, src_vocab, tgt_vocab = create_datasets(
        min_num=1, max_num=100000, sample_size=200, seed=42
    )

    # Show specific examples
    specific = [4000, 49999, 50000, 100000]
    print("\nSpecific examples:")
    for num in specific:
        roman = decimal_to_roman(num)
        back = roman_to_decimal(roman)
        print(f"  {num:>6} -> {roman:<20} -> {back} (round-trip: {'OK' if back == num else 'FAIL'})")

    # Show some random samples
    print("\nRandom samples from dataset:")
    for i in range(10):
        sample = train_ds[i]
        print(f"  {sample['decimal_str']:>6} -> {sample['roman_str']}")
