"""
Visualization functions for training analysis and results presentation.

Implements:
- Training curves (loss, accuracy)
- Attention heatmaps
- Data statistics plots
- Error analysis visualizations
- Confusion matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import torch


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    train_loss_std: Optional[List[float]] = None,
    val_loss_std: Optional[List[float]] = None,
    train_acc_std: Optional[List[float]] = None,
    val_acc_std: Optional[List[float]] = None,
    title: str = "Training Progress",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves with optional std shading.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch (optional)
        val_accs: Validation accuracies per epoch (optional)
        train_loss_std: Std of training loss per epoch (optional)
        val_loss_std: Std of validation loss per epoch (optional)
        train_acc_std: Std of training accuracy per epoch (optional)
        val_acc_std: Std of validation accuracy per epoch (optional)
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    set_style()

    has_acc = train_accs is not None and val_accs is not None
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(14 if has_acc else 8, 5))

    if not has_acc:
        axes = [axes]

    epochs = np.arange(1, len(train_losses) + 1)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    # Add std shading for losses
    if train_loss_std is not None:
        train_loss_std = np.array(train_loss_std)
        axes[0].fill_between(epochs, train_losses - train_loss_std, train_losses + train_loss_std,
                              color='blue', alpha=0.2)
    if val_loss_std is not None:
        val_loss_std = np.array(val_loss_std)
        axes[0].fill_between(epochs, val_losses - val_loss_std, val_losses + val_loss_std,
                              color='red', alpha=0.2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves (shaded = ±1 std)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc:
        train_accs = np.array(train_accs)
        val_accs = np.array(val_accs)

        axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)

        # Add std shading for accuracies
        if train_acc_std is not None:
            train_acc_std = np.array(train_acc_std)
            axes[1].fill_between(epochs,
                                  np.clip(train_accs - train_acc_std, 0, 1),
                                  np.clip(train_accs + train_acc_std, 0, 1),
                                  color='blue', alpha=0.2)
        if val_acc_std is not None:
            val_acc_std = np.array(val_acc_std)
            axes[1].fill_between(epochs,
                                  np.clip(val_accs - val_acc_std, 0, 1),
                                  np.clip(val_accs + val_acc_std, 0, 1),
                                  color='red', alpha=0.2)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves (shaded = ±1 std)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_attention_matrix(
    attention_weights: List[torch.Tensor],
    src_tokens: List[str],
    tgt_tokens: List[str],
    title: str = "Attention Weights",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention heatmap for a single example.

    Args:
        attention_weights: List of attention tensors for each output step
        src_tokens: Source sequence tokens
        tgt_tokens: Target sequence tokens (output)
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    set_style()

    # Stack attention weights: [tgt_len, src_len]
    if isinstance(attention_weights[0], torch.Tensor):
        attn_matrix = torch.stack(attention_weights).numpy()
    else:
        attn_matrix = np.stack(attention_weights)

    # Trim to actual sequence lengths
    tgt_len = len(tgt_tokens)
    src_len = len(src_tokens)
    attn_matrix = attn_matrix[:tgt_len, :src_len]

    fig, ax = plt.subplots(figsize=(max(8, src_len * 0.8), max(6, tgt_len * 0.5)))

    sns.heatmap(
        attn_matrix,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='Blues',
        annot=True if src_len * tgt_len <= 50 else False,
        fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )

    ax.set_xlabel('Source (Decimal Digits)')
    ax.set_ylabel('Target (Roman Numerals)')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_multiple_attention(
    examples: List[Dict[str, Any]],
    ncols: int = 2,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention matrices for multiple examples.

    Args:
        examples: List of dicts with 'attention', 'src_tokens', 'tgt_tokens', 'title'
        ncols: Number of columns in subplot grid
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    set_style()

    n = len(examples)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, ex in enumerate(examples):
        attn = ex['attention']
        if isinstance(attn[0], torch.Tensor):
            attn_matrix = torch.stack(attn).numpy()
        else:
            attn_matrix = np.stack(attn)

        src_len = len(ex['src_tokens'])
        tgt_len = len(ex['tgt_tokens'])
        attn_matrix = attn_matrix[:tgt_len, :src_len]

        sns.heatmap(
            attn_matrix,
            xticklabels=ex['src_tokens'],
            yticklabels=ex['tgt_tokens'],
            cmap='Blues',
            ax=axes[i],
            cbar=False
        )
        axes[i].set_title(ex.get('title', f'Example {i+1}'))
        axes[i].set_xlabel('Source')
        axes[i].set_ylabel('Target')

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_data_statistics(
    stats: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot dataset statistics.

    Args:
        stats: Dictionary from get_dataset_statistics()
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    set_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Source length distribution
    ax = axes[0, 0]
    src_lengths = stats['src_lengths']
    ax.hist(src_lengths, bins=range(1, max(src_lengths) + 2), edgecolor='black', alpha=0.7)
    ax.set_xlabel('Source Length (digits)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Source Sequence Lengths')
    ax.set_xticks(range(1, max(src_lengths) + 1))

    # Target length distribution
    ax = axes[0, 1]
    tgt_lengths = stats['tgt_lengths']
    ax.hist(tgt_lengths, bins=range(1, max(tgt_lengths) + 2), edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Target Length (Roman numerals)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Target Sequence Lengths')

    # Digit frequency
    ax = axes[1, 0]
    digit_freq = stats['digit_freq']
    digits = sorted(digit_freq.keys())
    counts = [digit_freq[d] for d in digits]
    ax.bar(digits, counts, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Frequency')
    ax.set_title('Digit Frequency in Source Sequences')

    # Roman numeral frequency
    ax = axes[1, 1]
    roman_freq = stats['roman_freq']
    romans = sorted(roman_freq.keys(), key=lambda x: 'IVXLCDM'.index(x) if x in 'IVXLCDM' else 100)
    counts = [roman_freq[r] for r in romans]
    ax.bar(romans, counts, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Roman Numeral')
    ax.set_ylabel('Frequency')
    ax.set_title('Roman Numeral Frequency in Target Sequences')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_error_matrices(
    confusion_matrices: Dict[int, np.ndarray],
    vocab: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrices for different positions.

    Args:
        confusion_matrices: Dict mapping position to confusion matrix
        vocab: Vocabulary characters
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    set_style()

    positions = sorted(confusion_matrices.keys())[:4]  # Show first 4 positions
    n = len(positions)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    labels = vocab + ['∅']  # ∅ for no character

    for i, pos in enumerate(positions):
        matrix = confusion_matrices[pos]

        # Normalize by row (true labels)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix_norm = matrix / row_sums

        sns.heatmap(
            matrix_norm,
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues',
            ax=axes[i],
            annot=False,
            cbar=True
        )
        axes[i].set_title(f'Position {pos + 1}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.suptitle('Confusion Matrices by Position (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_position_errors(
    position_stats: Dict[int, Tuple[int, int]],
    title: str = "Accuracy by Position",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot accuracy at each position.

    Args:
        position_stats: Dict mapping position to (correct, total)
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    set_style()

    positions = sorted(position_stats.keys())
    accuracies = [position_stats[p][0] / position_stats[p][1] if position_stats[p][1] > 0 else 0
                  for p in positions]
    totals = [position_stats[p][1] for p in positions]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Accuracy bars
    bars = ax1.bar([p + 1 for p in positions], accuracies, alpha=0.7, label='Accuracy')
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1)

    # Sample count line
    ax2 = ax1.twinx()
    ax2.plot([p + 1 for p in positions], totals, 'r-o', label='Sample Count', linewidth=2)
    ax2.set_ylabel('Sample Count', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.annotate(f'{acc:.2%}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    ax1.set_title(title)
    ax1.set_xticks([p + 1 for p in positions])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metric_names: List[str],
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare metrics across different models/configurations.

    Args:
        results: Dict mapping model name to metrics dict
        metric_names: List of metric names to plot
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    set_style()

    models = list(results.keys())
    n_metrics = len(metric_names)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 6))

    x = np.arange(n_models)
    width = 0.8 / n_metrics

    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for i, metric in enumerate(metric_names):
        values = [results[m].get(metric, 0) for m in models]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i])

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_error_by_length(
    error_analysis: Dict[int, Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot error metrics grouped by input length.

    Args:
        error_analysis: Dict from analyze_errors_by_length()
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    set_style()

    lengths = sorted(error_analysis.keys())
    seq_acc = [error_analysis[l]['seq_accuracy'] for l in lengths]
    char_acc = [error_analysis[l]['char_accuracy'] for l in lengths]
    lev_dist = [error_analysis[l]['mean_levenshtein'] for l in lengths]
    counts = [error_analysis[l]['count'] for l in lengths]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sequence accuracy by length
    ax = axes[0]
    bars = ax.bar(lengths, seq_acc, alpha=0.7)
    ax.set_xlabel('Number of Digits')
    ax.set_ylabel('Sequence Accuracy')
    ax.set_title('Exact Match Accuracy by Input Length')
    ax.set_ylim(0, 1)
    ax.set_xticks(lengths)
    for bar, acc in zip(bars, seq_acc):
        ax.annotate(f'{acc:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    # Character accuracy by length
    ax = axes[1]
    bars = ax.bar(lengths, char_acc, alpha=0.7, color='orange')
    ax.set_xlabel('Number of Digits')
    ax.set_ylabel('Character Accuracy')
    ax.set_title('Character-level Accuracy by Input Length')
    ax.set_ylim(0, 1)
    ax.set_xticks(lengths)

    # Levenshtein distance by length
    ax = axes[2]
    bars = ax.bar(lengths, lev_dist, alpha=0.7, color='green')
    ax.set_xlabel('Number of Digits')
    ax.set_ylabel('Mean Levenshtein Distance')
    ax.set_title('Mean Edit Distance by Input Length')
    ax.set_xticks(lengths)

    plt.suptitle(f'Error Analysis by Input Length (n={sum(counts)})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_decoding_comparison(
    decoding_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare different decoding strategies.

    Args:
        decoding_results: Dict mapping strategy name to metrics
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    return plot_metrics_comparison(
        decoding_results,
        ['seq_accuracy', 'char_accuracy', 'mean_normalized_levenshtein'],
        title="Decoding Strategy Comparison",
        save_path=save_path
    )
