#!/usr/bin/env python3
"""Evaluation script for Audio-Visual Event Localization."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AVEventDataset, create_dataloader
from src.models.av_event_localization import AVEventLocalizationModel
from src.losses.av_losses import AVEventLoss
from src.eval.metrics import AVEventMetrics
from src.utils.training import setup_device, load_checkpoint


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


def create_model(cfg: DictConfig) -> AVEventLocalizationModel:
    """Create the audio-visual event localization model."""
    from src.models.encoders import AudioEncoder, VisualEncoder
    from src.models.fusion import CrossAttentionFusion
    from src.models.temporal import TemporalTransformer
    from src.models.classifiers import EventClassifier
    
    # Create encoders
    audio_encoder = AudioEncoder(
        model_name=cfg.model.audio_encoder.model_name,
        freeze_encoder=cfg.model.audio_encoder.freeze_encoder,
        output_dim=cfg.model.audio_encoder.output_dim,
    )
    
    visual_encoder = VisualEncoder(
        model_name=cfg.model.visual_encoder.model_name,
        pretrained=cfg.model.visual_encoder.pretrained,
        freeze_backbone=cfg.model.visual_encoder.freeze_backbone,
        output_dim=cfg.model.visual_encoder.output_dim,
    )
    
    # Create fusion module
    fusion = CrossAttentionFusion(
        audio_dim=cfg.model.fusion.audio_dim,
        visual_dim=cfg.model.fusion.visual_dim,
        hidden_dim=cfg.model.fusion.hidden_dim,
        num_heads=cfg.model.fusion.num_heads,
        num_layers=cfg.model.fusion.num_layers,
        dropout=cfg.model.fusion.dropout,
    )
    
    # Create temporal encoder
    temporal_encoder = TemporalTransformer(
        input_dim=cfg.model.temporal_encoder.input_dim,
        hidden_dim=cfg.model.temporal_encoder.hidden_dim,
        num_heads=cfg.model.temporal_encoder.num_heads,
        num_layers=cfg.model.temporal_encoder.num_layers,
        dropout=cfg.model.temporal_encoder.dropout,
    )
    
    # Create event classifier
    event_classifier = EventClassifier(
        input_dim=cfg.model.event_classifier.input_dim,
        num_classes=cfg.model.event_classifier.num_classes,
        dropout=cfg.model.event_classifier.dropout,
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


def evaluate_model(
    model: AVEventLocalizationModel,
    dataloader: DataLoader,
    criterion: AVEventLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on the given dataset."""
    model.eval()
    metrics = AVEventMetrics()
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(audio, video)
            
            # Compute loss
            loss_dict = criterion(outputs, labels, outputs['audio_features'], outputs['visual_features'])
            loss = loss_dict['total_loss']
            
            # Update metrics
            predictions = torch.argmax(outputs['logits'], dim=-1)
            metrics.update(predictions, labels)
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
    
    # Compute final metrics
    final_metrics = metrics.compute()
    final_metrics['loss'] = total_loss / num_batches
    
    return final_metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Setup
    setup_logging(cfg.logging.level)
    device = setup_device(cfg.device)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create model
    model = create_model(cfg)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = cfg.get('checkpoint_path', 'checkpoints/best_model.pt')
    if Path(checkpoint_path).exists():
        checkpoint = load_checkpoint(model, filepath=checkpoint_path, device=device)
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, using random weights")
    
    # Create dataset
    dataset = AVEventDataset(
        data_dir=cfg.data_dir,
        split=cfg.get('split', 'test'),
        audio_sample_rate=cfg.data.audio_sample_rate,
        video_fps=cfg.data.video_fps,
        max_audio_length=cfg.data.max_audio_length,
        max_video_length=cfg.data.max_video_length,
    )
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    
    logger.info(f"Evaluating on {len(dataset)} samples")
    
    # Create loss function
    criterion = AVEventLoss(
        num_classes=cfg.model.event_classifier.num_classes,
        classification_weight=cfg.train.loss_weights.classification,
        alignment_weight=cfg.train.loss_weights.temporal_alignment,
        contrastive_weight=cfg.train.loss_weights.contrastive,
    )
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader, criterion, device)
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
    logger.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    
    if 'temporal_iou' in metrics:
        logger.info(f"Temporal IoU: {metrics['temporal_iou']:.4f}")
    
    if 'sync_error_mae' in metrics:
        logger.info(f"Sync Error (MAE): {metrics['sync_error_mae']:.4f}")
    
    # Save results
    results_path = Path(cfg.output_dir) / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
