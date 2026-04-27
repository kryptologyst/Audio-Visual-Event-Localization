"""Audio-Visual Event Localization Model.

This module implements a modern deep learning model for audio-visual event localization,
combining audio and visual encoders with cross-attention fusion and temporal modeling.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, AutoProcessor
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from .encoders import AudioEncoder, VisualEncoder
from .fusion import CrossAttentionFusion
from .temporal import TemporalTransformer
from .classifiers import EventClassifier


class AVEventLocalizationModel(nn.Module):
    """Audio-Visual Event Localization Model.
    
    This model combines audio and visual features to localize events in time and space.
    It uses cross-attention fusion to align audio and visual modalities and temporal
    transformers to model temporal dependencies.
    
    Args:
        audio_encoder: Audio feature encoder
        visual_encoder: Visual feature encoder  
        fusion: Cross-attention fusion module
        temporal_encoder: Temporal modeling module
        event_classifier: Event classification head
    """
    
    def __init__(
        self,
        audio_encoder: AudioEncoder,
        visual_encoder: VisualEncoder,
        fusion: CrossAttentionFusion,
        temporal_encoder: TemporalTransformer,
        event_classifier: EventClassifier,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.fusion = fusion
        self.temporal_encoder = temporal_encoder
        self.event_classifier = event_classifier
        
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            audio: Audio tensor of shape (batch_size, audio_length)
            video: Video tensor of shape (batch_size, num_frames, channels, height, width)
            audio_mask: Optional audio attention mask
            video_mask: Optional video attention mask
            
        Returns:
            Dictionary containing:
                - logits: Event classification logits
                - audio_features: Audio features
                - visual_features: Visual features
                - fused_features: Fused audio-visual features
                - temporal_features: Temporal features
        """
        # Encode audio and visual features
        audio_features = self.audio_encoder(audio, audio_mask)
        visual_features = self.visual_encoder(video, video_mask)
        
        # Fuse audio and visual features
        fused_features = self.fusion(audio_features, visual_features)
        
        # Apply temporal modeling
        temporal_features = self.temporal_encoder(fused_features)
        
        # Classify events
        logits = self.event_classifier(temporal_features)
        
        return {
            "logits": logits,
            "audio_features": audio_features,
            "visual_features": visual_features,
            "fused_features": fused_features,
            "temporal_features": temporal_features,
        }
    
    def get_attention_weights(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization.
        
        Args:
            audio: Audio tensor
            video: Video tensor
            audio_mask: Optional audio attention mask
            video_mask: Optional video attention mask
            
        Returns:
            Dictionary containing attention weights
        """
        # Encode features
        audio_features = self.audio_encoder(audio, audio_mask)
        visual_features = self.visual_encoder(video, video_mask)
        
        # Get fusion attention weights
        fused_features, fusion_attention = self.fusion(
            audio_features, visual_features, return_attention=True
        )
        
        # Get temporal attention weights
        temporal_features, temporal_attention = self.temporal_encoder(
            fused_features, return_attention=True
        )
        
        return {
            "fusion_attention": fusion_attention,
            "temporal_attention": temporal_attention,
        }
    
    def predict_events(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        threshold: float = 0.5,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict events with confidence scores.
        
        Args:
            audio: Audio tensor
            video: Video tensor
            threshold: Confidence threshold for predictions
            audio_mask: Optional audio attention mask
            video_mask: Optional video attention mask
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        with torch.no_grad():
            outputs = self.forward(audio, video, audio_mask, video_mask)
            logits = outputs["logits"]
            probabilities = F.softmax(logits, dim=-1)
            
            # Get predictions above threshold
            max_probs, predicted_classes = torch.max(probabilities, dim=-1)
            predictions = (max_probs > threshold).long()
            
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "confidence": max_probs,
                "predicted_classes": predicted_classes,
            }
