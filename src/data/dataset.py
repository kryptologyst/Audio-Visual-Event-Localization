"""Data loading and preprocessing for audio-visual event localization."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T


class AVEventDataset(Dataset):
    """Audio-Visual Event Localization Dataset.
    
    This dataset loads audio and video data for event localization tasks.
    
    Args:
        data_dir: Directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        audio_sample_rate: Audio sample rate
        video_fps: Video frame rate
        max_audio_length: Maximum audio length in seconds
        max_video_length: Maximum video length in seconds
        transform: Optional data augmentation transforms
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        audio_sample_rate: int = 16000,
        video_fps: int = 30,
        max_audio_length: float = 10.0,
        max_video_length: float = 10.0,
        transform: Optional[Dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_sample_rate = audio_sample_rate
        self.video_fps = video_fps
        self.max_audio_length = max_audio_length
        self.max_video_length = max_video_length
        self.transform = transform or {}
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Audio preprocessing transforms
        self.audio_transform = T.MelSpectrogram(
            sample_rate=audio_sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
        )
        
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations."""
        annotation_file = self.data_dir / f"annotations_{self.split}.json"
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                return json.load(f)
        else:
            # Create synthetic annotations for demo
            return self._create_synthetic_annotations()
    
    def _create_synthetic_annotations(self) -> List[Dict]:
        """Create synthetic annotations for demonstration."""
        synthetic_data = []
        
        # Create a few synthetic samples
        for i in range(10):
            sample = {
                "id": f"synthetic_{i}",
                "audio_path": f"audio/sample_{i}.wav",
                "video_path": f"video/sample_{i}.mp4",
                "events": [
                    {
                        "start_time": 1.0 + i * 0.5,
                        "end_time": 2.0 + i * 0.5,
                        "event_type": i % 3,  # 3 event types
                        "confidence": 0.8 + (i % 3) * 0.1,
                    }
                ],
                "duration": 5.0,
            }
            synthetic_data.append(sample)
        
        return synthetic_data
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing audio, video, and labels
        """
        annotation = self.annotations[idx]
        
        # Load audio
        audio_path = self.data_dir / annotation["audio_path"]
        audio = self._load_audio(audio_path)
        
        # Load video
        video_path = self.data_dir / annotation["video_path"]
        video = self._load_video(video_path)
        
        # Create labels
        labels = self._create_labels(annotation)
        
        # Apply transforms
        if self.transform:
            audio, video = self._apply_transforms(audio, video)
        
        return {
            "audio": audio,
            "video": video,
            "labels": labels,
            "sample_id": annotation["id"],
            "duration": annotation["duration"],
        }
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and preprocess audio."""
        if audio_path.exists():
            # Load real audio file
            audio, sr = librosa.load(str(audio_path), sr=self.audio_sample_rate)
        else:
            # Create synthetic audio
            duration = self.max_audio_length
            audio = np.random.randn(int(duration * self.audio_sample_rate))
        
        # Convert to tensor
        audio = torch.from_numpy(audio).float()
        
        # Pad or truncate to max length
        max_length = int(self.max_audio_length * self.audio_sample_rate)
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = F.pad(audio, (0, max_length - len(audio)))
        
        return audio
    
    def _load_video(self, video_path: Path) -> torch.Tensor:
        """Load and preprocess video."""
        if video_path.exists():
            # Load real video file
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                # Create synthetic video if loading failed
                frames = self._create_synthetic_video()
        else:
            # Create synthetic video
            frames = self._create_synthetic_video()
        
        # Convert to tensor and normalize
        video = torch.from_numpy(np.array(frames)).float()
        video = video.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        video = video / 255.0  # Normalize to [0, 1]
        
        # Resize frames
        video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Pad or truncate to max frames
        max_frames = int(self.max_video_length * self.video_fps)
        if video.shape[0] > max_frames:
            video = video[:max_frames]
        else:
            padding = max_frames - video.shape[0]
            video = F.pad(video, (0, 0, 0, 0, 0, padding))
        
        return video
    
    def _create_synthetic_video(self) -> List[np.ndarray]:
        """Create synthetic video frames."""
        frames = []
        num_frames = int(self.max_video_length * self.video_fps)
        
        for i in range(num_frames):
            # Create a simple synthetic frame
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            frames.append(frame)
        
        return frames
    
    def _create_labels(self, annotation: Dict) -> torch.Tensor:
        """Create labels from annotation."""
        # Create temporal labels
        duration = annotation["duration"]
        num_frames = int(duration * self.video_fps)
        
        # Initialize labels (0 = no event)
        labels = torch.zeros(num_frames, dtype=torch.long)
        
        # Set event labels
        for event in annotation["events"]:
            start_frame = int(event["start_time"] * self.video_fps)
            end_frame = int(event["end_time"] * self.video_fps)
            event_type = event["event_type"] + 1  # +1 because 0 is no event
            
            labels[start_frame:end_frame] = event_type
        
        return labels
    
    def _apply_transforms(self, audio: torch.Tensor, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation transforms."""
        # Audio augmentation
        if "audio_noise" in self.transform:
            noise_factor = self.transform["audio_noise"]
            noise = torch.randn_like(audio) * noise_factor
            audio = audio + noise
        
        if "audio_time_shift" in self.transform:
            shift_factor = self.transform["audio_time_shift"]
            shift = int(len(audio) * shift_factor)
            audio = torch.roll(audio, shift)
        
        # Video augmentation
        if "video_horizontal_flip" in self.transform and np.random.random() < self.transform["video_horizontal_flip"]:
            video = torch.flip(video, dims=[3])  # Flip width dimension
        
        if "video_color_jitter" in self.transform:
            jitter_factor = self.transform["video_color_jitter"]
            # Simple color jitter
            video = video + torch.randn_like(video) * jitter_factor
            video = torch.clamp(video, 0, 1)
        
        return audio, video


def create_dataloader(
    dataset: AVEventDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        dataset: The dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching samples.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched data dictionary
    """
    audio = torch.stack([sample["audio"] for sample in batch])
    video = torch.stack([sample["video"] for sample in batch])
    labels = torch.stack([sample["labels"] for sample in batch])
    
    return {
        "audio": audio,
        "video": video,
        "labels": labels,
        "sample_ids": [sample["sample_id"] for sample in batch],
        "durations": torch.tensor([sample["duration"] for sample in batch]),
    }
