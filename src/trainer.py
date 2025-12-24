"""
Training utilities for Seq2Seq models.

Implements:
- Trainer class with training loop
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
import time
from pathlib import Path
from tqdm import tqdm
import json

from .models import Seq2Seq
from .decoding import greedy_decode
from .metrics import calculate_metrics
from .data import Vocabulary


class Trainer:
    """
    Trainer for Seq2Seq models with attention.
    """

    def __init__(
        self,
        model: Seq2Seq,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.0,
        optimizer_type: str = 'adam',
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[Dict] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Seq2Seq model
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            grad_clip: Gradient clipping threshold
            label_smoothing: Label smoothing for cross-entropy loss
            optimizer_type: 'adam', 'adamw', or 'sgd'
            scheduler_type: 'step', 'cosine', 'plateau', or None
            scheduler_params: Parameters for scheduler
        """
        self.model = model.to(device)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.grad_clip = grad_clip

        # Loss function with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tgt_vocab.pad_idx,
            label_smoothing=label_smoothing
        )

        # Optimizer
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Scheduler
        self.scheduler = None
        if scheduler_type:
            scheduler_params = scheduler_params or {}
            if scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_params.get('step_size', 10),
                    gamma=scheduler_params.get('gamma', 0.5)
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_params.get('T_max', 50)
                )
            elif scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_params.get('factor', 0.5),
                    patience=scheduler_params.get('patience', 5)
                )

        # Training history (mean and std per epoch)
        self.history = {
            'train_loss': [],
            'train_loss_std': [],
            'val_loss': [],
            'val_loss_std': [],
            'train_seq_acc': [],
            'train_seq_acc_std': [],
            'val_seq_acc': [],
            'val_seq_acc_std': [],
            'train_char_acc': [],
            'val_char_acc': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[float, float, Dict[str, float], float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            Tuple of (average loss, loss std, metrics dict, seq_acc std)
        """
        import numpy as np

        self.model.train()
        batch_losses = []
        batch_seq_accs = []
        all_predictions = []
        all_targets = []

        pbar = tqdm(train_loader, desc='Training', leave=False)

        for batch in pbar:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_lengths = batch['src_len'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs, _ = self.model(src, src_lengths, tgt, teacher_forcing_ratio)

            # Compute loss (outputs: [batch, tgt_len-1, vocab], tgt: [batch, tgt_len])
            # We compare outputs to tgt[:, 1:] (excluding SOS)
            output_dim = outputs.shape[-1]
            outputs_flat = outputs.reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = self.criterion(outputs_flat, tgt_flat)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            batch_losses.append(loss.item())

            # Get predictions for metrics (batch-level)
            pred_indices = outputs.argmax(dim=-1)
            batch_correct = 0
            for i in range(src.size(0)):
                pred = self._decode_indices(pred_indices[i])
                target = batch['roman_str'][i]
                all_predictions.append(pred)
                all_targets.append(target)
                if pred == target:
                    batch_correct += 1

            batch_seq_accs.append(batch_correct / src.size(0))
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = np.mean(batch_losses)
        loss_std = np.std(batch_losses)
        seq_acc_std = np.std(batch_seq_accs)
        metrics = calculate_metrics(all_predictions, all_targets)

        return avg_loss, loss_std, metrics, seq_acc_std

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        use_greedy: bool = True
    ) -> Tuple[float, float, Dict[str, float], float, List[str], List[str]]:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: Data loader
            use_greedy: Whether to use greedy decoding (vs teacher forcing)

        Returns:
            Tuple of (average loss, loss std, metrics dict, seq_acc std, predictions, targets)
        """
        import numpy as np

        self.model.eval()
        batch_losses = []
        batch_seq_accs = []
        all_predictions = []
        all_targets = []
        all_decimals = []

        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_lengths = batch['src_len'].to(self.device)

            if use_greedy:
                # Use greedy decoding
                max_len = tgt.size(1)
                pred_indices, _ = greedy_decode(
                    self.model, src, src_lengths, max_len,
                    self.tgt_vocab.sos_idx, self.tgt_vocab.eos_idx
                )

                batch_correct = 0
                for i in range(src.size(0)):
                    pred = self.tgt_vocab.decode(pred_indices[i])
                    all_predictions.append(pred)
                    if pred == batch['roman_str'][i]:
                        batch_correct += 1
                batch_seq_accs.append(batch_correct / src.size(0))

                # Compute loss with teacher forcing for consistency
                outputs, _ = self.model(src, src_lengths, tgt, teacher_forcing_ratio=1.0)
            else:
                outputs, _ = self.model(src, src_lengths, tgt, teacher_forcing_ratio=1.0)
                pred_indices = outputs.argmax(dim=-1)

                batch_correct = 0
                for i in range(src.size(0)):
                    pred = self._decode_indices(pred_indices[i])
                    all_predictions.append(pred)
                    if pred == batch['roman_str'][i]:
                        batch_correct += 1
                batch_seq_accs.append(batch_correct / src.size(0))

            # Compute loss
            output_dim = outputs.shape[-1]
            outputs_flat = outputs.reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = self.criterion(outputs_flat, tgt_flat)
            batch_losses.append(loss.item())

            # Collect targets
            for i in range(src.size(0)):
                all_targets.append(batch['roman_str'][i])
                all_decimals.append(batch['decimal'][i].item())

        avg_loss = np.mean(batch_losses)
        loss_std = np.std(batch_losses)
        seq_acc_std = np.std(batch_seq_accs)
        metrics = calculate_metrics(all_predictions, all_targets)

        return avg_loss, loss_std, metrics, seq_acc_std, all_predictions, all_targets

    def _decode_indices(self, indices: torch.Tensor) -> str:
        """Decode tensor indices to string, stopping at EOS."""
        tokens = []
        for idx in indices.tolist():
            if idx == self.tgt_vocab.eos_idx:
                break
            if idx not in [self.tgt_vocab.pad_idx, self.tgt_vocab.sos_idx]:
                tokens.append(self.tgt_vocab.idx2token[idx])
        return ''.join(tokens)

    @torch.no_grad()
    def show_examples(
        self,
        data_loader: DataLoader,
        n_examples: int = 5
    ) -> List[Dict[str, str]]:
        """
        Show sample predictions from the model.

        Args:
            data_loader: Data loader to sample from
            n_examples: Number of examples to show

        Returns:
            List of dicts with 'input', 'target', 'prediction', 'correct'
        """
        self.model.eval()
        examples = []

        for batch in data_loader:
            src = batch['src'].to(self.device)
            src_lengths = batch['src_len'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            max_len = tgt.size(1)

            # Get predictions using greedy decoding
            pred_indices, _ = greedy_decode(
                self.model, src, src_lengths, max_len,
                self.tgt_vocab.sos_idx, self.tgt_vocab.eos_idx
            )

            for i in range(min(src.size(0), n_examples - len(examples))):
                decimal_str = batch['decimal_str'][i]
                target = batch['roman_str'][i]
                prediction = self.tgt_vocab.decode(pred_indices[i])
                correct = prediction == target

                examples.append({
                    'input': decimal_str,
                    'target': target,
                    'prediction': prediction,
                    'correct': correct
                })

                if len(examples) >= n_examples:
                    break

            if len(examples) >= n_examples:
                break

        return examples

    def print_examples(self, examples: List[Dict[str, str]]):
        """Print example predictions in a formatted way."""
        print("  Sample predictions:")
        for ex in examples:
            status = "OK" if ex['correct'] else "X"
            print(f"    [{status}] {ex['input']} -> {ex['prediction']} (target: {ex['target']})")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        teacher_forcing_ratio: float = 1.0,
        teacher_forcing_decay: float = 0.0,
        early_stopping_patience: int = 0,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
        show_examples: int = 0
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            teacher_forcing_ratio: Initial teacher forcing ratio
            teacher_forcing_decay: Decay rate for teacher forcing per epoch
            early_stopping_patience: Stop if no improvement for N epochs (0 = disabled)
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to print progress
            show_examples: Number of example predictions to show after each epoch (0 = disabled)

        Returns:
            Training history
        """

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Decay teacher forcing
            tf_ratio = max(0.0, teacher_forcing_ratio - teacher_forcing_decay * (epoch - 1))

            # Train
            train_loss, train_loss_std, train_metrics, train_acc_std = self.train_epoch(train_loader, tf_ratio)

            # Evaluate
            val_loss, val_loss_std, val_metrics, val_acc_std, _, _ = self.evaluate(val_loader)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history (mean and std)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_loss_std'].append(train_loss_std)
            self.history['val_loss'].append(val_loss)
            self.history['val_loss_std'].append(val_loss_std)
            self.history['train_seq_acc'].append(train_metrics['seq_accuracy'])
            self.history['train_seq_acc_std'].append(train_acc_std)
            self.history['val_seq_acc'].append(val_metrics['seq_accuracy'])
            self.history['val_seq_acc_std'].append(val_acc_std)
            self.history['train_char_acc'].append(train_metrics['char_accuracy'])
            self.history['val_char_acc'].append(val_metrics['char_accuracy'])
            self.history['learning_rate'].append(current_lr)

            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True
                self.epochs_without_improvement = 0

                # Save best model
                if checkpoint_dir:
                    self.save_checkpoint(
                        Path(checkpoint_dir) / 'best_model.pt',
                        epoch, val_loss, val_metrics
                    )
            else:
                self.epochs_without_improvement += 1

            if val_metrics['seq_accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['seq_accuracy']

            # Print progress
            elapsed = time.time() - start_time
            if verbose:
                print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
                      f"TF: {tf_ratio:.2f} | LR: {current_lr:.2e} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_metrics['seq_accuracy']:.2%} | "
                      f"Val Acc: {val_metrics['seq_accuracy']:.2%}"
                      f"{' *' if improved else ''}")

            # Show example predictions
            if show_examples > 0 and verbose:
                examples = self.show_examples(val_loader, n_examples=show_examples)
                self.print_examples(examples)

            # Early stopping
            if early_stopping_patience > 0 and self.epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
                break

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        val_metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'history': self.history
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
