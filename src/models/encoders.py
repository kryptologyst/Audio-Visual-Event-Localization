"""Audio and visual encoder modules."""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
    """Audio encoder using Wav2Vec2.
    
    This encoder extracts audio features using a pre-trained Wav2Vec2 model.
    
    Args:
        model_name: Name of the Wav2Vec2 model to use
        freeze_encoder: Whether to freeze the encoder weights
        output_dim: Output feature dimension
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze_encoder: bool = False,
        output_dim: int = 768,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load pre-trained Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Projection layer to match output dimension
        self.projection = nn.Linear(
            self.wav2vec2.config.hidden_size, output_dim
        )
        
    def forward(
        self, 
        audio: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the audio encoder.
        
        Args:
            audio: Audio tensor of shape (batch_size, audio_length)
            attention_mask: Optional attention mask
            
        Returns:
            Audio features of shape (batch_size, sequence_length, output_dim)
        """
        # Get Wav2Vec2 features
        outputs = self.wav2vec2(audio, attention_mask=attention_mask)
        features = outputs.last_hidden_state
        
        # Project to output dimension
        features = self.projection(features)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim


class VisualEncoder(nn.Module):
    """Visual encoder using ResNet50.
    
    This encoder extracts visual features from video frames using a pre-trained ResNet50.
    
    Args:
        model_name: Name of the visual model to use
        pretrained: Whether to use pre-trained weights
        freeze_backbone: Whether to freeze the backbone weights
        output_dim: Output feature dimension
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 2048,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load pre-trained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection layer to match output dimension
        self.projection = nn.Linear(2048, output_dim)
        
    def forward(
        self, 
        video: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the visual encoder.
        
        Args:
            video: Video tensor of shape (batch_size, num_frames, channels, height, width)
            attention_mask: Optional attention mask
            
        Returns:
            Visual features of shape (batch_size, num_frames, output_dim)
        """
        batch_size, num_frames, channels, height, width = video.shape
        
        # Reshape to process all frames at once
        video_flat = video.view(batch_size * num_frames, channels, height, width)
        
        # Extract features using ResNet50
        features = self.backbone(video_flat)
        features = features.view(batch_size * num_frames, -1)
        
        # Project to output dimension
        features = self.projection(features)
        
        # Reshape back to (batch_size, num_frames, output_dim)
        features = features.view(batch_size, num_frames, self.output_dim)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim
