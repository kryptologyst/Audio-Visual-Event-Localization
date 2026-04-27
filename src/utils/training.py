"""Training utilities and device management."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device.
    
    Args:
        device: Preferred device ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        PyTorch device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


def setup_device(device: Optional[str] = None) -> torch.device:
    """Setup device with optimal settings.
    
    Args:
        device: Preferred device
        
    Returns:
        Configured PyTorch device
    """
    device = get_device(device)
    
    if device.type == "cuda":
        # Enable cuDNN optimizations
        cudnn.benchmark = True
        cudnn.enabled = True
        
        # Set memory growth
        torch.cuda.empty_cache()
        
    elif device.type == "mps":
        # MPS-specific optimizations
        torch.mps.empty_cache()
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """Get human-readable model size.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size as string
    """
    param_count = count_parameters(model)
    
    if param_count < 1e3:
        return f"{param_count:.0f} parameters"
    elif param_count < 1e6:
        return f"{param_count/1e3:.1f}K parameters"
    elif param_count < 1e9:
        return f"{param_count/1e6:.1f}M parameters"
    else:
        return f"{param_count/1e9:.1f}B parameters"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: dict,
    filepath: str,
    **kwargs
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    filepath: str = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint data
    """
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights for
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
            
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
