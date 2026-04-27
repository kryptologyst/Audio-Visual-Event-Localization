#!/usr/bin/env python3
"""Training script for Audio-Visual Event Localization."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AVEventDataset, create_dataloader
from src.models.av_event_localization import AVEventLocalizationModel
from src.losses.av_losses import AVEventLoss
from src.eval.metrics import AVEventMetrics
from src.utils.training import (
    set_seed, setup_device, save_checkpoint, load_checkpoint, 
    EarlyStopping, count_parameters
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def create_model(cfg: DictConfig) -> AVEventLocalizationModel:
    """Create the audio-visual event localization model.
    
    Args:
        cfg: Model configuration
        
    Returns:
        Initialized model
    """
    # Import model components
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


def create_datasets(cfg: DictConfig) -> Dict[str, AVEventDataset]:
    """Create training and validation datasets.
    
    Args:
        cfg: Data configuration
        
    Returns:
        Dictionary containing train and val datasets
    """
    datasets = {}
    
    # Training dataset
    datasets['train'] = AVEventDataset(
        data_dir=cfg.data_dir,
        split='train',
        audio_sample_rate=cfg.data.audio_sample_rate,
        video_fps=cfg.data.video_fps,
        max_audio_length=cfg.data.max_audio_length,
        max_video_length=cfg.data.max_video_length,
        transform=cfg.data.get('audio_augment', {}),
    )
    
    # Validation dataset
    datasets['val'] = AVEventDataset(
        data_dir=cfg.data_dir,
        split='val',
        audio_sample_rate=cfg.data.audio_sample_rate,
        video_fps=cfg.data.video_fps,
        max_audio_length=cfg.data.max_audio_length,
        max_video_length=cfg.data.max_video_length,
        transform={},  # No augmentation for validation
    )
    
    return datasets


def create_dataloaders(
    datasets: Dict[str, AVEventDataset], 
    cfg: DictConfig
) -> Dict[str, DataLoader]:
    """Create data loaders.
    
    Args:
        datasets: Dictionary of datasets
        cfg: Data configuration
        
    Returns:
        Dictionary containing train and val data loaders
    """
    dataloaders = {}
    
    dataloaders['train'] = create_dataloader(
        datasets['train'],
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    
    dataloaders['val'] = create_dataloader(
        datasets['val'],
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    
    return dataloaders


def train_epoch(
    model: AVEventLocalizationModel,
    dataloader: DataLoader,
    criterion: AVEventLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    metrics = AVEventMetrics()
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(audio, video)
        
        # Compute loss
        loss_dict = criterion(outputs, labels, outputs['audio_features'], outputs['visual_features'])
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        predictions = torch.argmax(outputs['logits'], dim=-1)
        metrics.update(predictions, labels)
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
        })
    
    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = total_loss / num_batches
    
    return epoch_metrics


def validate_epoch(
    model: AVEventLocalizationModel,
    dataloader: DataLoader,
    criterion: AVEventLoss,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    metrics = AVEventMetrics()
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        
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
    
    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    epoch_metrics['loss'] = total_loss / num_batches
    
    return epoch_metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup
    set_seed(cfg.seed)
    setup_logging(cfg.logging.level)
    device = setup_device(cfg.device)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create model
    model = create_model(cfg)
    model = model.to(device)
    
    logger.info(f"Model created with {count_parameters(model)} parameters")
    
    # Create datasets and dataloaders
    datasets = create_datasets(cfg)
    dataloaders = create_dataloaders(datasets, cfg)
    
    logger.info(f"Training samples: {len(datasets['train'])}")
    logger.info(f"Validation samples: {len(datasets['val'])}")
    
    # Create loss function
    criterion = AVEventLoss(
        num_classes=cfg.model.event_classifier.num_classes,
        classification_weight=cfg.train.loss_weights.classification,
        alignment_weight=cfg.train.loss_weights.temporal_alignment,
        contrastive_weight=cfg.train.loss_weights.contrastive,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=10,
        mode='max',
        restore_best_weights=True,
    )
    
    # Training loop
    best_val_f1 = 0.0
    
    for epoch in range(cfg.train.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{cfg.train.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch + 1
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, dataloaders['val'], criterion, device, epoch + 1
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics.get('f1_macro', 0):.4f}")
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics.get('f1_macro', 0):.4f}")
        
        # Save checkpoint
        if epoch % cfg.train.save_every_n_epochs == 0:
            checkpoint_path = Path(cfg.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics['loss'], val_metrics, str(checkpoint_path)
            )
        
        # Early stopping
        val_f1 = val_metrics.get('f1_macro', 0)
        if early_stopping(val_f1, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save best model
            best_model_path = Path(cfg.checkpoint_dir) / "best_model.pt"
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics['loss'], val_metrics, str(best_model_path)
            )
    
    logger.info(f"Training completed. Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
