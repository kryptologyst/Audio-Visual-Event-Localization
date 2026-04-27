"""Evaluation metrics for audio-visual event localization."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class AVEventMetrics:
    """Evaluation metrics for audio-visual event localization.
    
    This class computes various metrics including accuracy, F1-score,
    temporal IoU, and audio-visual synchronization metrics.
    """
    
    def __init__(self, num_classes: int = 10, temporal_tolerance: float = 0.5):
        self.num_classes = num_classes
        self.temporal_tolerance = temporal_tolerance
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.temporal_predictions = []
        self.temporal_targets = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        temporal_preds: Optional[List[Dict]] = None,
        temporal_targets: Optional[List[Dict]] = None,
    ):
        """Update metrics with new predictions.
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth labels
            confidences: Prediction confidence scores
            temporal_preds: Temporal predictions
            temporal_targets: Temporal ground truth
        """
        # Convert to numpy for sklearn metrics
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        self.predictions.extend(preds_np.flatten())
        self.targets.extend(targets_np.flatten())
        
        if confidences is not None:
            conf_np = confidences.cpu().numpy()
            self.confidences.extend(conf_np.flatten())
        
        if temporal_preds is not None:
            self.temporal_predictions.extend(temporal_preds)
        
        if temporal_targets is not None:
            self.temporal_targets.extend(temporal_targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        if not self.predictions:
            return metrics
        
        # Convert to numpy arrays
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Classification metrics
        metrics["accuracy"] = accuracy_score(targets, preds)
        metrics["f1_macro"] = f1_score(targets, preds, average='macro', zero_division=0)
        metrics["f1_micro"] = f1_score(targets, preds, average='micro', zero_division=0)
        metrics["precision_macro"] = precision_score(targets, preds, average='macro', zero_division=0)
        metrics["recall_macro"] = recall_score(targets, preds, average='macro', zero_division=0)
        
        # Per-class metrics
        for i in range(self.num_classes):
            class_mask = targets == i
            if np.sum(class_mask) > 0:
                class_preds = preds[class_mask]
                class_targets = targets[class_mask]
                metrics[f"f1_class_{i}"] = f1_score(class_targets, class_preds, average='binary', zero_division=0)
        
        # Temporal metrics
        if self.temporal_predictions and self.temporal_targets:
            temporal_metrics = self._compute_temporal_metrics()
            metrics.update(temporal_metrics)
        
        # Confidence metrics
        if self.confidences:
            confidence_metrics = self._compute_confidence_metrics()
            metrics.update(confidence_metrics)
        
        return metrics
    
    def _compute_temporal_metrics(self) -> Dict[str, float]:
        """Compute temporal evaluation metrics."""
        metrics = {}
        
        if not self.temporal_predictions or not self.temporal_targets:
            return metrics
        
        ious = []
        sync_errors = []
        
        for pred, target in zip(self.temporal_predictions, self.temporal_targets):
            # Compute temporal IoU
            iou = self._compute_temporal_iou(pred, target)
            ious.append(iou)
            
            # Compute synchronization error
            sync_error = self._compute_sync_error(pred, target)
            sync_errors.append(sync_error)
        
        metrics["temporal_iou"] = np.mean(ious) if ious else 0.0
        metrics["sync_error_mae"] = np.mean(sync_errors) if sync_errors else 0.0
        
        return metrics
    
    def _compute_temporal_iou(self, pred: Dict, target: Dict) -> float:
        """Compute temporal IoU between prediction and target."""
        pred_start = pred.get("start_time", 0)
        pred_end = pred.get("end_time", 0)
        target_start = target.get("start_time", 0)
        target_end = target.get("end_time", 0)
        
        # Compute intersection
        intersection_start = max(pred_start, target_start)
        intersection_end = min(pred_end, target_end)
        intersection = max(0, intersection_end - intersection_start)
        
        # Compute union
        union = (pred_end - pred_start) + (target_end - target_start) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_sync_error(self, pred: Dict, target: Dict) -> float:
        """Compute audio-visual synchronization error."""
        pred_center = (pred.get("start_time", 0) + pred.get("end_time", 0)) / 2
        target_center = (target.get("start_time", 0) + target.get("end_time", 0)) / 2
        
        return abs(pred_center - target_center)
    
    def _compute_confidence_metrics(self) -> Dict[str, float]:
        """Compute confidence-based metrics."""
        metrics = {}
        
        if not self.confidences:
            return metrics
        
        confidences = np.array(self.confidences)
        targets = np.array(self.targets)
        predictions = np.array(self.predictions)
        
        # Confidence calibration
        correct_mask = predictions == targets
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        metrics["confidence_correct_mean"] = np.mean(correct_conf) if len(correct_conf) > 0 else 0.0
        metrics["confidence_incorrect_mean"] = np.mean(incorrect_conf) if len(incorrect_conf) > 0 else 0.0
        metrics["confidence_gap"] = metrics["confidence_correct_mean"] - metrics["confidence_incorrect_mean"]
        
        return metrics


def compute_audio_visual_sync_metric(
    audio_features: torch.Tensor,
    visual_features: torch.Tensor,
    tolerance: float = 0.5,
) -> float:
    """Compute audio-visual synchronization metric.
    
    Args:
        audio_features: Audio features
        visual_features: Visual features
        tolerance: Temporal tolerance in seconds
        
    Returns:
        Synchronization score
    """
    # Compute cross-correlation between audio and visual features
    audio_norm = F.normalize(audio_features.mean(dim=1), dim=-1)
    visual_norm = F.normalize(visual_features.mean(dim=1), dim=-1)
    
    # Compute similarity
    similarity = torch.sum(audio_norm * visual_norm, dim=-1)
    
    # Convert to synchronization score
    sync_score = torch.mean(similarity).item()
    
    return sync_score


def compute_temporal_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    tolerance_frames: int = 15,  # ~0.5 seconds at 30fps
) -> float:
    """Compute temporal accuracy with tolerance.
    
    Args:
        predictions: Predicted temporal labels
        targets: Ground truth temporal labels
        tolerance_frames: Temporal tolerance in frames
        
    Returns:
        Temporal accuracy
    """
    batch_size, seq_len = predictions.shape
    
    correct = 0
    total = 0
    
    for b in range(batch_size):
        pred_seq = predictions[b]
        target_seq = targets[b]
        
        # Find event boundaries
        pred_events = self._find_event_boundaries(pred_seq)
        target_events = self._find_event_boundaries(target_seq)
        
        # Match events with tolerance
        matched = 0
        for pred_event in pred_events:
            for target_event in target_events:
                if abs(pred_event - target_event) <= tolerance_frames:
                    matched += 1
                    break
        
        correct += matched
        total += len(target_events)
    
    return correct / total if total > 0 else 0.0


def _find_event_boundaries(sequence: torch.Tensor) -> List[int]:
    """Find event boundaries in a sequence."""
    boundaries = []
    
    # Find transitions
    diff = torch.diff(sequence)
    change_points = torch.where(diff != 0)[0]
    
    for point in change_points:
        boundaries.append(point.item())
    
    return boundaries
