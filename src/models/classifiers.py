"""Event classification module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventClassifier(nn.Module):
    """Event classification head.
    
    This module classifies events from temporal features.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of event classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, num_classes),
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the event classifier.
        
        Args:
            features: Input features of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Classification logits of shape (batch_size, seq_len, num_classes)
        """
        return self.classifier(features)
