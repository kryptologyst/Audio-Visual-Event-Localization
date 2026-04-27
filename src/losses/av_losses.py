"""Loss functions for audio-visual event localization."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVEventLoss(nn.Module):
    """Combined loss for audio-visual event localization.
    
    This loss combines classification loss, temporal alignment loss,
    and contrastive loss for multi-modal learning.
    
    Args:
        num_classes: Number of event classes
        classification_weight: Weight for classification loss
        alignment_weight: Weight for temporal alignment loss
        contrastive_weight: Weight for contrastive loss
        temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        classification_weight: float = 1.0,
        alignment_weight: float = 0.5,
        contrastive_weight: float = 0.3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classification_weight = classification_weight
        self.alignment_weight = alignment_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.alignment_loss = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss(temperature)
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth labels
            audio_features: Audio features for contrastive loss
            visual_features: Visual features for contrastive loss
            
        Returns:
            Dictionary containing individual and total losses
        """
        logits = predictions["logits"]
        batch_size, seq_len, num_classes = logits.shape
        
        # Reshape for classification loss
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        # Classification loss
        classification_loss = self.classification_loss(logits_flat, targets_flat)
        
        # Temporal alignment loss (encourage audio-visual synchronization)
        alignment_loss = 0.0
        if self.alignment_weight > 0 and "fused_features" in predictions:
            fused_features = predictions["fused_features"]
            # Simple alignment: encourage similar features at same time steps
            audio_feat = predictions.get("audio_features", torch.zeros_like(fused_features))
            visual_feat = predictions.get("visual_features", torch.zeros_like(fused_features))
            
            # Compute feature similarity
            audio_norm = F.normalize(audio_feat, dim=-1)
            visual_norm = F.normalize(visual_feat, dim=-1)
            similarity = torch.sum(audio_norm * visual_norm, dim=-1)
            
            # Encourage high similarity (alignment)
            alignment_loss = self.alignment_loss(similarity, torch.ones_like(similarity))
        
        # Contrastive loss
        contrastive_loss = 0.0
        if self.contrastive_weight > 0 and audio_features is not None and visual_features is not None:
            contrastive_loss = self.contrastive_loss(audio_features, visual_features)
        
        # Total loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.alignment_weight * alignment_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "alignment_loss": alignment_loss,
            "contrastive_loss": contrastive_loss,
        }


class ContrastiveLoss(nn.Module):
    """Contrastive loss for audio-visual alignment."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            audio_features: Audio features
            visual_features: Visual features
            
        Returns:
            Contrastive loss
        """
        batch_size = audio_features.shape[0]
        
        # Normalize features
        audio_norm = F.normalize(audio_features.mean(dim=1), dim=-1)  # Average over time
        visual_norm = F.normalize(visual_features.mean(dim=1), dim=-1)  # Average over time
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(audio_norm, visual_norm.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size).to(audio_features.device)
        
        # Compute loss
        loss_audio_to_visual = F.cross_entropy(similarity_matrix, labels)
        loss_visual_to_audio = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_audio_to_visual + loss_visual_to_audio) / 2


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
