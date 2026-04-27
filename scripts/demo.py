#!/usr/bin/env python3
"""Demo script for Audio-Visual Event Localization.

This script demonstrates the modernized audio-visual event localization system
with synthetic data and shows the key capabilities.
"""

import logging
import numpy as np
import torch
from pathlib import Path

from src.models.encoders import AudioEncoder, VisualEncoder
from src.models.fusion import CrossAttentionFusion
from src.models.temporal import TemporalTransformer
from src.models.classifiers import EventClassifier
from src.models.av_event_localization import AVEventLocalizationModel
from src.losses.av_losses import AVEventLoss
from src.eval.metrics import AVEventMetrics
from src.utils.training import set_seed, setup_device


def create_demo_model():
    """Create a demo model with default parameters."""
    # Create encoders
    audio_encoder = AudioEncoder(
        model_name="facebook/wav2vec2-base",
        freeze_encoder=False,
        output_dim=768,
    )
    
    visual_encoder = VisualEncoder(
        model_name="resnet50",
        pretrained=True,
        freeze_backbone=False,
        output_dim=2048,
    )
    
    # Create fusion module
    fusion = CrossAttentionFusion(
        audio_dim=768,
        visual_dim=2048,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    )
    
    # Create temporal encoder
    temporal_encoder = TemporalTransformer(
        input_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )
    
    # Create event classifier
    event_classifier = EventClassifier(
        input_dim=512,
        num_classes=10,
        dropout=0.1,
    )
    
    # Create main model
    model = AVEventLocalizationModel(
        audio_encoder=audio_encoder,
        visual_encoder=visual_encoder,
        fusion=fusion,
        temporal_encoder=temporal_encoder,
        event_classifier=event_classifier,
    )
    
    return model


def create_demo_data():
    """Create synthetic demo data."""
    batch_size = 2
    audio_length = 16000  # 1 second at 16kHz
    num_frames = 30  # 1 second at 30fps
    height, width = 224, 224
    channels = 3
    
    # Create synthetic audio
    audio = torch.randn(batch_size, audio_length)
    
    # Create synthetic video
    video = torch.randn(batch_size, num_frames, channels, height, width)
    
    # Create synthetic labels
    labels = torch.randint(0, 10, (batch_size, num_frames))
    
    return audio, video, labels


def main():
    """Main demo function."""
    # Setup
    set_seed(42)
    device = setup_device("auto")
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎵 Audio-Visual Event Localization Demo")
    logger.info("=" * 50)
    
    # Create model
    logger.info("Creating model...")
    model = create_demo_model()
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created successfully!")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create demo data
    logger.info("\nCreating demo data...")
    audio, video, labels = create_demo_data()
    audio = audio.to(device)
    video = video.to(device)
    labels = labels.to(device)
    
    logger.info(f"Audio shape: {audio.shape}")
    logger.info(f"Video shape: {video.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    
    # Forward pass
    logger.info("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(audio, video)
    
    logger.info("Forward pass completed!")
    logger.info(f"Logits shape: {outputs['logits'].shape}")
    logger.info(f"Audio features shape: {outputs['audio_features'].shape}")
    logger.info(f"Visual features shape: {outputs['visual_features'].shape}")
    logger.info(f"Fused features shape: {outputs['fused_features'].shape}")
    logger.info(f"Temporal features shape: {outputs['temporal_features'].shape}")
    
    # Test loss computation
    logger.info("\nTesting loss computation...")
    criterion = AVEventLoss(
        num_classes=10,
        classification_weight=1.0,
        alignment_weight=0.5,
        contrastive_weight=0.3,
    )
    
    loss_dict = criterion(outputs, labels, outputs['audio_features'], outputs['visual_features'])
    
    logger.info("Loss computation completed!")
    logger.info(f"Total loss: {loss_dict['total_loss']:.4f}")
    logger.info(f"Classification loss: {loss_dict['classification_loss']:.4f}")
    logger.info(f"Alignment loss: {loss_dict['alignment_loss']:.4f}")
    logger.info(f"Contrastive loss: {loss_dict['contrastive_loss']:.4f}")
    
    # Test metrics
    logger.info("\nTesting evaluation metrics...")
    metrics = AVEventMetrics(num_classes=10)
    
    predictions = torch.argmax(outputs['logits'], dim=-1)
    metrics.update(predictions, labels)
    
    metric_results = metrics.compute()
    
    logger.info("Metrics computation completed!")
    logger.info(f"Accuracy: {metric_results['accuracy']:.4f}")
    logger.info(f"F1-Score (Macro): {metric_results['f1_macro']:.4f}")
    logger.info(f"F1-Score (Micro): {metric_results['f1_micro']:.4f}")
    logger.info(f"Precision (Macro): {metric_results['precision_macro']:.4f}")
    logger.info(f"Recall (Macro): {metric_results['recall_macro']:.4f}")
    
    # Test event prediction
    logger.info("\nTesting event prediction...")
    predictions = model.predict_events(audio, video, threshold=0.5)
    
    logger.info("Event prediction completed!")
    logger.info(f"Predictions shape: {predictions['predictions'].shape}")
    logger.info(f"Confidence shape: {predictions['confidence'].shape}")
    logger.info(f"Predicted classes shape: {predictions['predicted_classes'].shape}")
    
    # Show sample predictions
    for i in range(min(2, predictions['predictions'].shape[0])):
        num_events = predictions['predictions'][i].sum().item()
        avg_confidence = predictions['confidence'][i].mean().item()
        logger.info(f"Sample {i}: {num_events} events detected, avg confidence: {avg_confidence:.3f}")
    
    # Test attention weights
    logger.info("\nTesting attention weight extraction...")
    attention_weights = model.get_attention_weights(audio, video)
    
    logger.info("Attention weights extracted!")
    logger.info(f"Fusion attention keys: {list(attention_weights['fusion_attention'][0].keys())}")
    
    logger.info("\n" + "=" * 50)
    logger.info("✅ Demo completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Run 'python scripts/train.py' to train the model")
    logger.info("2. Run 'streamlit run demo/app.py' to launch the interactive demo")
    logger.info("3. Run 'python scripts/evaluate.py' to evaluate a trained model")
    logger.info("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()
