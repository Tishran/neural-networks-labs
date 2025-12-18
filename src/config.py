"""
Experiment configuration management.

Provides dataclasses for configuring experiments and utilities for
saving/loading configurations.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for dataset."""
    min_num: int = 1
    max_num: int = 3999  # Standard Roman range (use >3999 for vinculum notation)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    max_src_len: Optional[int] = None
    max_tgt_len: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    embed_size: int = 64
    hidden_size: int = 128
    num_layers: int = 1
    cell_type: str = 'lstm'  # 'lstm' or 'gru'
    bidirectional: bool = False
    attention_method: str = 'general'  # 'dot', 'general', 'concat'
    dropout: float = 0.1
    embedding_type: str = 'learned'  # 'learned' or 'onehot'


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    optimizer_type: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler_type: Optional[str] = None  # 'step', 'cosine', 'plateau'
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    teacher_forcing_ratio: float = 1.0
    teacher_forcing_decay: float = 0.0
    early_stopping_patience: int = 2


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'seed': self.seed
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(
            name=d.get('name', 'default'),
            data=DataConfig(**d.get('data', {})),
            model=ModelConfig(**d.get('model', {})),
            training=TrainingConfig(**d.get('training', {})),
            seed=d.get('seed', 42)
        )

    def save(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def get_model_name(self) -> str:
        """Generate descriptive model name from config."""
        m = self.model
        parts = [
            m.cell_type.upper(),
            'Bi' if m.bidirectional else 'Uni',
            f'L{m.num_layers}',
            f'H{m.hidden_size}',
            f'E{m.embed_size}',
            m.attention_method
        ]
        return '_'.join(parts)


# Predefined experiment configurations
def get_baseline_config() -> ExperimentConfig:
    """Get baseline experiment configuration."""
    return ExperimentConfig(
        name="baseline",
        model=ModelConfig(
            embed_size=64,
            hidden_size=128,
            num_layers=1,
            cell_type='lstm',
            bidirectional=False,
            attention_method='general',
            dropout=0.1,
            embedding_type='learned'
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=50,
            learning_rate=1e-3,
            early_stopping_patience=3
        )
    )


def get_experiment_configs() -> Dict[str, ExperimentConfig]:
    """
    Get all experiment configurations for the lab.

    Returns dictionary mapping experiment name to config.
    """
    configs = {}

    # Baseline
    configs['baseline'] = get_baseline_config()

    # 1. Varying sequence length (data range experiments)
    for max_num in [99, 999, 3999]:
        name = f"range_{max_num}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.data.max_num = max_num
        configs[name] = cfg

    # 2. Architecture experiments - hidden sizes
    for hidden in [64, 128, 256, 512]:
        name = f"hidden_{hidden}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.model.hidden_size = hidden
        configs[name] = cfg

    # 3. Architecture experiments - number of layers
    for layers in [1, 2, 3]:
        name = f"layers_{layers}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.model.num_layers = layers
        if layers > 1:
            cfg.model.dropout = 0.2
        configs[name] = cfg

    # 4. Architecture experiments - embedding size
    for embed in [32, 64, 128]:
        name = f"embed_{embed}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.model.embed_size = embed
        configs[name] = cfg

    # 5. Bidirectional vs Unidirectional
    configs['unidirectional'] = get_baseline_config()
    configs['unidirectional'].name = 'unidirectional'

    cfg = get_baseline_config()
    cfg.name = 'bidirectional'
    cfg.model.bidirectional = True
    configs['bidirectional'] = cfg

    # 6. LSTM vs GRU
    configs['lstm'] = get_baseline_config()
    configs['lstm'].name = 'lstm'

    cfg = get_baseline_config()
    cfg.name = 'gru'
    cfg.model.cell_type = 'gru'
    configs['gru'] = cfg

    # 7. Embedding types
    configs['learned_embed'] = get_baseline_config()
    configs['learned_embed'].name = 'learned_embed'

    cfg = get_baseline_config()
    cfg.name = 'onehot_embed'
    cfg.model.embedding_type = 'onehot'
    configs['onehot_embed'] = cfg

    # 8. Attention methods
    for method in ['dot', 'general', 'concat']:
        name = f"attention_{method}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.model.attention_method = method
        configs[name] = cfg

    # 9. Regularization - dropout
    for dropout in [0.0, 0.1, 0.2, 0.3, 0.5]:
        name = f"dropout_{dropout}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.model.dropout = dropout
        configs[name] = cfg

    # 10. Regularization - weight decay (L2)
    for wd in [0.0, 1e-5, 1e-4, 1e-3]:
        name = f"weight_decay_{wd}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.training.weight_decay = wd
        configs[name] = cfg

    # 11. Label smoothing
    for ls in [0.0, 0.05, 0.1, 0.2]:
        name = f"label_smooth_{ls}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.training.label_smoothing = ls
        configs[name] = cfg

    # 12. Learning rate
    for lr in [5e-4, 1e-3, 2e-3, 5e-3]:
        name = f"lr_{lr}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.training.learning_rate = lr
        configs[name] = cfg

    # 13. Optimizers
    for opt in ['adam', 'adamw', 'sgd']:
        name = f"optimizer_{opt}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.training.optimizer_type = opt
        configs[name] = cfg

    # 14. Teacher forcing decay
    for decay in [0.0, 0.01, 0.02]:
        name = f"tf_decay_{decay}"
        cfg = get_baseline_config()
        cfg.name = name
        cfg.training.teacher_forcing_decay = decay
        configs[name] = cfg

    # 15. Large model configuration
    cfg = ExperimentConfig(
        name="large_model",
        model=ModelConfig(
            embed_size=128,
            hidden_size=256,
            num_layers=2,
            cell_type='lstm',
            bidirectional=True,
            attention_method='general',
            dropout=0.2,
            embedding_type='learned'
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=100,
            learning_rate=1e-3,
            early_stopping_patience=3
        )
    )
    configs['large_model'] = cfg

    return configs


def get_decoding_configs() -> List[Dict[str, Any]]:
    """Get configurations for different decoding strategies."""
    return [
        {'strategy': 'greedy'},
        {'strategy': 'beam', 'beam_size': 3},
        {'strategy': 'beam', 'beam_size': 5},
        {'strategy': 'beam', 'beam_size': 10},
        {'strategy': 'top_k', 'k': 3, 'temperature': 1.0},
        {'strategy': 'top_k', 'k': 5, 'temperature': 1.0},
        {'strategy': 'top_k', 'k': 5, 'temperature': 0.7},
        {'strategy': 'top_p', 'p': 0.9, 'temperature': 1.0},
        {'strategy': 'top_p', 'p': 0.95, 'temperature': 1.0},
        {'strategy': 'top_p', 'p': 0.9, 'temperature': 0.7},
    ]
