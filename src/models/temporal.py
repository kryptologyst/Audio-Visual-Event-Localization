"""Temporal modeling module using transformers."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    """Temporal transformer for modeling temporal dependencies.
    
    This module uses a transformer encoder to model temporal relationships
    in the fused audio-visual features.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for the transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the temporal transformer.
        
        Args:
            features: Input features of shape (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Temporal features of shape (batch_size, seq_len, hidden_dim)
        """
        # Project to hidden dimension
        features = self.input_projection(features)
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Apply transformer
        if return_attention:
            # For attention visualization, we need to modify the transformer
            # This is a simplified version - in practice, you might want to
            # implement custom attention extraction
            temporal_features = self.transformer(features, src_key_padding_mask=attention_mask)
            attention_weights = None  # Would need custom implementation for full attention
        else:
            temporal_features = self.transformer(features, src_key_padding_mask=attention_mask)
            attention_weights = None
        
        # Project to output dimension
        temporal_features = self.output_projection(temporal_features)
        
        if return_attention:
            return temporal_features, attention_weights
        return temporal_features


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
