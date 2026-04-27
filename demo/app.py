"""Streamlit demo for Audio-Visual Event Localization."""

import io
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import librosa
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Audio-Visual Event Localization",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Safety disclaimer
DISCLAIMER = """
**DISCLAIMER**: This model is for research and educational purposes only. 
Not intended for surveillance or privacy-invasive applications. 
Users must comply with applicable privacy laws and regulations.
"""


def load_model() -> Optional[torch.nn.Module]:
    """Load the trained model."""
    try:
        # In a real implementation, you would load your trained model here
        # For demo purposes, we'll create a mock model
        st.info("Loading model... (Demo mode - using mock model)")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def process_audio(audio_file) -> Tuple[np.ndarray, int]:
    """Process uploaded audio file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Load audio
        audio, sr = librosa.load(tmp_file_path, sr=16000)
        
        # Clean up
        Path(tmp_file_path).unlink()
        
        return audio, sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None


def process_video(video_file) -> Tuple[List[np.ndarray], int]:
    """Process uploaded video file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file_path = tmp_file.name
        
        # Load video frames
        cap = cv2.VideoCapture(tmp_file_path)
        frames = []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Clean up
        Path(tmp_file_path).unlink()
        
        return frames, fps
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, None


def create_synthetic_data() -> Tuple[np.ndarray, List[np.ndarray]]:
    """Create synthetic audio-visual data for demo."""
    # Create synthetic audio (10 seconds at 16kHz)
    duration = 10.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create audio with events
    audio = np.zeros_like(t)
    audio += 0.1 * np.sin(2 * np.pi * 440 * t)  # Base tone
    
    # Add events
    event_times = [2.0, 5.0, 8.0]
    for event_time in event_times:
        start_idx = int(event_time * sample_rate)
        end_idx = int((event_time + 1.0) * sample_rate)
        audio[start_idx:end_idx] += 0.3 * np.sin(2 * np.pi * 880 * t[start_idx:end_idx])
    
    # Add noise
    audio += 0.05 * np.random.randn(len(audio))
    
    # Create synthetic video frames
    num_frames = int(duration * 30)  # 30 fps
    frames = []
    
    for i in range(num_frames):
        # Create a simple frame with moving elements
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add moving rectangle for events
        if 2.0 <= i / 30 <= 3.0 or 5.0 <= i / 30 <= 6.0 or 8.0 <= i / 30 <= 9.0:
            cv2.rectangle(frame, (50, 50), (150, 150), (255, 0, 0), -1)
        
        frames.append(frame)
    
    return audio, frames


def visualize_audio(audio: np.ndarray, sample_rate: int) -> go.Figure:
    """Create audio visualization."""
    time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio,
        mode='lines',
        name='Audio Waveform',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        showlegend=False
    )
    
    return fig


def visualize_video_frames(frames: List[np.ndarray], fps: int) -> List[Image.Image]:
    """Convert video frames to PIL Images for display."""
    images = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)
        images.append(img)
    return images


def predict_events(audio: np.ndarray, frames: List[np.ndarray]) -> Dict:
    """Mock event prediction (replace with actual model inference)."""
    # This is a mock prediction - in reality, you would use your trained model
    duration = len(audio) / 16000  # Assuming 16kHz sample rate
    
    # Mock predictions
    events = []
    event_types = ["Speech", "Music", "Noise", "Silence"]
    
    # Create some mock events
    for i in range(3):
        start_time = np.random.uniform(0, duration - 2)
        end_time = start_time + np.random.uniform(0.5, 2.0)
        event_type = np.random.choice(event_types)
        confidence = np.random.uniform(0.6, 0.95)
        
        events.append({
            "start_time": start_time,
            "end_time": end_time,
            "event_type": event_type,
            "confidence": confidence
        })
    
    return {
        "events": events,
        "audio_features": np.random.randn(100, 768),  # Mock features
        "visual_features": np.random.randn(len(frames), 2048),  # Mock features
        "attention_weights": np.random.rand(10, 10),  # Mock attention
    }


def visualize_predictions(audio: np.ndarray, predictions: Dict, sample_rate: int) -> go.Figure:
    """Visualize event predictions on audio waveform."""
    time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))
    
    fig = go.Figure()
    
    # Plot audio waveform
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio,
        mode='lines',
        name='Audio Waveform',
        line=dict(color='lightblue', width=1)
    ))
    
    # Plot events
    colors = ['red', 'green', 'orange', 'purple']
    event_types = ["Speech", "Music", "Noise", "Silence"]
    
    for i, event in enumerate(predictions["events"]):
        color = colors[i % len(colors)]
        event_type = event["event_type"]
        
        # Add event rectangle
        fig.add_vrect(
            x0=event["start_time"],
            x1=event["end_time"],
            fillcolor=color,
            opacity=0.3,
            annotation_text=f"{event_type}<br>Conf: {event['confidence']:.2f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Audio-Visual Event Localization Results",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=400,
        showlegend=False
    )
    
    return fig


def visualize_attention(attention_weights: np.ndarray) -> go.Figure:
    """Visualize attention weights."""
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="Cross-Modal Attention Weights",
        xaxis_title="Visual Features",
        yaxis_title="Audio Features",
        height=400
    )
    
    return fig


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🎵 Audio-Visual Event Localization</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    max_events = st.sidebar.slider("Max Events to Display", 1, 10, 5)
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload Files", "Use Synthetic Data"]
    )
    
    # Safety disclaimer
    st.sidebar.markdown(f'<div class="warning-box">{DISCLAIMER}</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Data")
        
        if data_source == "Upload Files":
            # File upload
            audio_file = st.file_uploader(
                "Upload Audio File",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Upload an audio file for analysis"
            )
            
            video_file = st.file_uploader(
                "Upload Video File",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for analysis"
            )
            
            if audio_file and video_file:
                # Process uploaded files
                audio, sample_rate = process_audio(audio_file)
                frames, fps = process_video(video_file)
                
                if audio is not None and frames is not None:
                    st.success("Files uploaded and processed successfully!")
                else:
                    st.error("Error processing uploaded files")
                    return
            else:
                st.info("Please upload both audio and video files")
                return
                
        else:  # Synthetic data
            if st.button("Generate Synthetic Data"):
                with st.spinner("Generating synthetic data..."):
                    audio, frames = create_synthetic_data()
                    sample_rate = 16000
                    fps = 30
                st.success("Synthetic data generated!")
            else:
                st.info("Click 'Generate Synthetic Data' to create demo data")
                return
        
        # Display audio visualization
        if 'audio' in locals():
            st.subheader("Audio Visualization")
            audio_fig = visualize_audio(audio, sample_rate)
            st.plotly_chart(audio_fig, use_container_width=True)
        
        # Display video frames
        if 'frames' in locals():
            st.subheader("Video Frames")
            
            # Show frame slider
            frame_idx = st.slider("Select Frame", 0, len(frames) - 1, 0)
            
            if frames:
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                st.image(img, caption=f"Frame {frame_idx}", use_column_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        
        if 'audio' in locals() and 'frames' in locals():
            # Run prediction
            if st.button("Analyze Events", type="primary"):
                with st.spinner("Analyzing audio-visual events..."):
                    predictions = predict_events(audio, frames)
                
                # Display results
                st.subheader("Event Detection Results")
                
                # Create results table
                events_data = []
                for i, event in enumerate(predictions["events"]):
                    events_data.append({
                        "Event #": i + 1,
                        "Type": event["event_type"],
                        "Start Time": f"{event['start_time']:.2f}s",
                        "End Time": f"{event['end_time']:.2f}s",
                        "Duration": f"{event['end_time'] - event['start_time']:.2f}s",
                        "Confidence": f"{event['confidence']:.3f}"
                    })
                
                st.table(events_data)
                
                # Visualize predictions
                st.subheader("Event Timeline")
                pred_fig = visualize_predictions(audio, predictions, sample_rate)
                st.plotly_chart(pred_fig, use_container_width=True)
                
                # Attention visualization
                st.subheader("Attention Analysis")
                attn_fig = visualize_attention(predictions["attention_weights"])
                st.plotly_chart(attn_fig, use_container_width=True)
                
                # Metrics
                st.subheader("Performance Metrics")
                
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric("Events Detected", len(predictions["events"]))
                
                with col_metric2:
                    avg_confidence = np.mean([e["confidence"] for e in predictions["events"]])
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                
                with col_metric3:
                    total_duration = sum([e["end_time"] - e["start_time"] for e in predictions["events"]])
                    st.metric("Total Event Duration", f"{total_duration:.2f}s")
        
        else:
            st.info("Please provide input data to see analysis results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Audio-Visual Event Localization Demo** | "
        "Built with Streamlit | "
        "For research and educational purposes only"
    )


if __name__ == "__main__":
    main()
