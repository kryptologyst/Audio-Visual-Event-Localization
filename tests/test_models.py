"""Tests for audio-visual event localization."""

import pytest
import torch
import numpy as np

from src.models.encoders import AudioEncoder, VisualEncoder
from src.models.fusion import CrossAttentionFusion
from src.models.temporal import TemporalTransformer
from src.models.classifiers import EventClassifier
from src.models.av_event_localization import AVEventLocalizationModel
from src.losses.av_losses import AVEventLoss
from src.eval.metrics import AVEventMetrics
from src.utils.training import set_seed, get_device


class TestAudioEncoder:
    """Test audio encoder functionality."""
    
    def test_audio_encoder_creation(self):
        """Test audio encoder can be created."""
        encoder = AudioEncoder()
        assert encoder is not None
        assert encoder.get_feature_dim() == 768
    
    def test_audio_encoder_forward(self):
        """Test audio encoder forward pass."""
        encoder = AudioEncoder()
        batch_size = 2
        audio_length = 16000  # 1 second at 16kHz
        
        audio = torch.randn(batch_size, audio_length)
        features = encoder(audio)
        
        assert features.shape[0] == batch_size
        assert features.shape[-1] == 768


class TestVisualEncoder:
    """Test visual encoder functionality."""
    
    def test_visual_encoder_creation(self):
        """Test visual encoder can be created."""
        encoder = VisualEncoder()
        assert encoder is not None
        assert encoder.get_feature_dim() == 2048
    
    def test_visual_encoder_forward(self):
        """Test visual encoder forward pass."""
        encoder = VisualEncoder()
        batch_size = 2
        num_frames = 30
        height, width = 224, 224
        channels = 3
        
        video = torch.randn(batch_size, num_frames, channels, height, width)
        features = encoder(video)
        
        assert features.shape[0] == batch_size
        assert features.shape[1] == num_frames
        assert features.shape[2] == 2048


class TestCrossAttentionFusion:
    """Test cross-attention fusion functionality."""
    
    def test_fusion_creation(self):
        """Test fusion module can be created."""
        fusion = CrossAttentionFusion(
            audio_dim=768,
            visual_dim=2048,
            hidden_dim=512
        )
        assert fusion is not None
    
    def test_fusion_forward(self):
        """Test fusion module forward pass."""
        fusion = CrossAttentionFusion(
            audio_dim=768,
            visual_dim=2048,
            hidden_dim=512
        )
        
        batch_size = 2
        audio_seq_len = 100
        visual_seq_len = 30
        
        audio_features = torch.randn(batch_size, audio_seq_len, 768)
        visual_features = torch.randn(batch_size, visual_seq_len, 2048)
        
        fused_features = fusion(audio_features, visual_features)
        
        assert fused_features.shape[0] == batch_size
        assert fused_features.shape[-1] == 512


class TestTemporalTransformer:
    """Test temporal transformer functionality."""
    
    def test_temporal_transformer_creation(self):
        """Test temporal transformer can be created."""
        transformer = TemporalTransformer(input_dim=512)
        assert transformer is not None
    
    def test_temporal_transformer_forward(self):
        """Test temporal transformer forward pass."""
        transformer = TemporalTransformer(input_dim=512)
        
        batch_size = 2
        seq_len = 100
        
        features = torch.randn(batch_size, seq_len, 512)
        temporal_features = transformer(features)
        
        assert temporal_features.shape[0] == batch_size
        assert temporal_features.shape[1] == seq_len
        assert temporal_features.shape[2] == 512


class TestEventClassifier:
    """Test event classifier functionality."""
    
    def test_classifier_creation(self):
        """Test event classifier can be created."""
        classifier = EventClassifier(input_dim=512, num_classes=10)
        assert classifier is not None
    
    def test_classifier_forward(self):
        """Test event classifier forward pass."""
        classifier = EventClassifier(input_dim=512, num_classes=10)
        
        batch_size = 2
        seq_len = 100
        
        features = torch.randn(batch_size, seq_len, 512)
        logits = classifier(features)
        
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == seq_len
        assert logits.shape[2] == 10


class TestAVEventLocalizationModel:
    """Test main model functionality."""
    
    def test_model_creation(self):
        """Test main model can be created."""
        # Create components
        audio_encoder = AudioEncoder()
        visual_encoder = VisualEncoder()
        fusion = CrossAttentionFusion(768, 2048, 512)
        temporal_encoder = TemporalTransformer(512)
        event_classifier = EventClassifier(512, 10)
        
        # Create main model
        model = AVEventLocalizationModel(
            audio_encoder=audio_encoder,
            visual_encoder=visual_encoder,
            fusion=fusion,
            temporal_encoder=temporal_encoder,
            event_classifier=event_classifier,
        )
        
        assert model is not None
    
    def test_model_forward(self):
        """Test main model forward pass."""
        # Create components
        audio_encoder = AudioEncoder()
        visual_encoder = VisualEncoder()
        fusion = CrossAttentionFusion(768, 2048, 512)
        temporal_encoder = TemporalTransformer(512)
        event_classifier = EventClassifier(512, 10)
        
        # Create main model
        model = AVEventLocalizationModel(
            audio_encoder=audio_encoder,
            visual_encoder=visual_encoder,
            fusion=fusion,
            temporal_encoder=temporal_encoder,
            event_classifier=event_classifier,
        )
        
        batch_size = 2
        audio_length = 16000
        num_frames = 30
        
        audio = torch.randn(batch_size, audio_length)
        video = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        outputs = model(audio, video)
        
        assert "logits" in outputs
        assert "audio_features" in outputs
        assert "visual_features" in outputs
        assert "fused_features" in outputs
        assert "temporal_features" in outputs


class TestAVEventLoss:
    """Test loss function functionality."""
    
    def test_loss_creation(self):
        """Test loss function can be created."""
        loss_fn = AVEventLoss(num_classes=10)
        assert loss_fn is not None
    
    def test_loss_computation(self):
        """Test loss computation."""
        loss_fn = AVEventLoss(num_classes=10)
        
        batch_size = 2
        seq_len = 100
        
        # Mock predictions
        predictions = {
            "logits": torch.randn(batch_size, seq_len, 10),
            "audio_features": torch.randn(batch_size, seq_len, 768),
            "visual_features": torch.randn(batch_size, seq_len, 2048),
        }
        
        targets = torch.randint(0, 10, (batch_size, seq_len))
        
        loss_dict = loss_fn(predictions, targets)
        
        assert "total_loss" in loss_dict
        assert "classification_loss" in loss_dict
        assert "alignment_loss" in loss_dict
        assert "contrastive_loss" in loss_dict


class TestAVEventMetrics:
    """Test evaluation metrics functionality."""
    
    def test_metrics_creation(self):
        """Test metrics can be created."""
        metrics = AVEventMetrics(num_classes=10)
        assert metrics is not None
    
    def test_metrics_update_and_compute(self):
        """Test metrics update and computation."""
        metrics = AVEventMetrics(num_classes=10)
        
        batch_size = 2
        seq_len = 100
        
        predictions = torch.randint(0, 10, (batch_size, seq_len))
        targets = torch.randint(0, 10, (batch_size, seq_len))
        
        metrics.update(predictions, targets)
        results = metrics.compute()
        
        assert "accuracy" in results
        assert "f1_macro" in results
        assert "f1_micro" in results


class TestTrainingUtils:
    """Test training utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This should not raise an exception
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert isinstance(device, torch.device)


if __name__ == "__main__":
    pytest.main([__file__])
