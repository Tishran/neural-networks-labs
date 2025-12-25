# Neural Networks Lab 2: LSTM Encoder-Decoder with Attention
# Decimal to Roman Numeral Translation

from .data import (
    DecimalRomanDataset, create_datasets, create_dataloaders,
    get_dataset_statistics, decimal_to_roman, Vocabulary
)
from .models import Encoder, Decoder, Seq2Seq, create_model
from .attention import LuongAttention
from .metrics import (
    calculate_metrics, levenshtein_distance, position_wise_accuracy,
    confusion_matrix_for_position, analyze_errors_by_length, analyze_errors_by_range
)
from .decoding import (
    greedy_decode, beam_search, top_k_decode, top_p_decode, decode_with_strategy
)
from .visualization import (
    plot_training_curves, plot_training_curves_comparison,
    plot_attention_matrix, plot_multiple_attention,
    plot_data_statistics, plot_error_matrices, plot_position_errors,
    plot_metrics_comparison, plot_error_by_length, plot_decoding_comparison
)
from .trainer import Trainer, get_device, count_parameters
from .config import (
    ExperimentConfig, DataConfig, ModelConfig, TrainingConfig,
    get_baseline_config, get_experiment_configs, get_decoding_configs
)

__all__ = [
    # Data
    'DecimalRomanDataset', 'create_datasets', 'create_dataloaders',
    'get_dataset_statistics', 'decimal_to_roman', 'Vocabulary',
    # Models
    'Encoder', 'Decoder', 'Seq2Seq', 'create_model',
    # Attention
    'LuongAttention',
    # Metrics
    'calculate_metrics', 'levenshtein_distance', 'position_wise_accuracy',
    'confusion_matrix_for_position', 'analyze_errors_by_length', 'analyze_errors_by_range',
    # Decoding
    'greedy_decode', 'beam_search', 'top_k_decode', 'top_p_decode', 'decode_with_strategy',
    # Visualization
    'plot_training_curves', 'plot_training_curves_comparison',
    'plot_attention_matrix', 'plot_multiple_attention',
    'plot_data_statistics', 'plot_error_matrices', 'plot_position_errors',
    'plot_metrics_comparison', 'plot_error_by_length', 'plot_decoding_comparison',
    # Trainer
    'Trainer', 'get_device', 'count_parameters',
    # Config
    'ExperimentConfig', 'DataConfig', 'ModelConfig', 'TrainingConfig',
    'get_baseline_config', 'get_experiment_configs', 'get_decoding_configs',
]
