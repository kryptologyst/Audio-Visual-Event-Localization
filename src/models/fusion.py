"""Cross-attention fusion module for audio-visual features."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module.
    
    This module fuses audio and visual features using cross-attention mechanisms.
    
    Args:
        audio_dim: Audio feature dimension
        visual_dim: Visual feature dimension
        hidden_dim: Hidden dimension for attention
        num_heads: Number of attention heads
        num_layers: Number of attention layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Projection layers to match hidden dimension
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the cross-attention fusion.
        
        Args:
            audio_features: Audio features of shape (batch_size, audio_seq_len, audio_dim)
            visual_features: Visual features of shape (batch_size, visual_seq_len, visual_dim)
            audio_mask: Optional audio attention mask
            visual_mask: Optional visual attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Fused features of shape (batch_size, max_seq_len, hidden_dim)
        """
        # Project features to hidden dimension
        audio_proj = self.audio_projection(audio_features)
        visual_proj = self.visual_projection(visual_features)
        
        # Apply cross-attention layers
        audio_out = audio_proj
        visual_out = visual_proj
        attention_weights = []
        
        for layer in self.cross_attention_layers:
            audio_out, visual_out, attn_weights = layer(
                audio_out, visual_out, audio_mask, visual_mask, return_attention=True
            )
            attention_weights.append(attn_weights)
        
        # Concatenate audio and visual features
        fused_features = torch.cat([audio_out, visual_out], dim=-1)
        
        # Project to output dimension
        fused_features = self.output_projection(fused_features)
        
        if return_attention:
            return fused_features, attention_weights
        return fused_features


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.audio_to_visual_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_to_audio_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.audio_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.visual_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the cross-attention layer.
        
        Args:
            audio_features: Audio features
            visual_features: Visual features
            audio_mask: Optional audio attention mask
            visual_mask: Optional visual attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (updated_audio_features, updated_visual_features, attention_weights)
        """
        # Audio attends to visual features
        audio_attended, audio_attn_weights = self.audio_to_visual_attention(
            audio_features, visual_features, visual_features,
            key_padding_mask=visual_mask,
            need_weights=return_attention,
        )
        audio_features = self.audio_norm(audio_features + audio_attended)
        audio_features = audio_features + self.audio_ffn(audio_features)
        
        # Visual attends to audio features
        visual_attended, visual_attn_weights = self.visual_to_audio_attention(
            visual_features, audio_features, audio_features,
            key_padding_mask=audio_mask,
            need_weights=return_attention,
        )
        visual_features = self.visual_norm(visual_features + visual_attended)
        visual_features = visual_features + self.visual_ffn(visual_features)
        
        if return_attention:
            attention_weights = {
                "audio_to_visual": audio_attn_weights,
                "visual_to_audio": visual_attn_weights,
            }
            return audio_features, visual_features, attention_weights
        
        return audio_features, visual_features, None
