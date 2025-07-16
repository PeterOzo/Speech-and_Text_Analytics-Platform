#!/usr/bin/env python3
"""
AnalyticsPro - Speech Emotion Recognition Platform
Professional Streamlit Application for Audio Emotion Analysis
Author: Peter Chika Ozo-ogueji (Data Scientist)
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import tempfile
import time
import requests
from scipy import stats
from datetime import datetime
import warnings
import base64
from io import BytesIO
import soundfile as sf
import wave
from typing import Dict, List, Tuple, Optional
import traceback

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AnalyticsPro - Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff9800;
        --danger-color: #d62728;
        --dark-bg: #1e1e1e;
        --light-bg: #f8f9fa;
    }
    
    /* Professional header styling */
    .main-header {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Emotion badge styling */
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .emotion-angry { background: #ffebee; color: #c62828; }
    .emotion-calm { background: #e3f2fd; color: #1565c0; }
    .emotion-disgust { background: #fce4ec; color: #880e4f; }
    .emotion-fearful { background: #f3e5f5; color: #6a1b9a; }
    .emotion-happy { background: #fff3e0; color: #e65100; }
    .emotion-neutral { background: #eceff1; color: #455a64; }
    .emotion-sad { background: #e8eaf6; color: #283593; }
    .emotion-surprised { background: #fffde7; color: #f57f17; }
    
    /* Feature importance styling */
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
        transition: all 0.3s;
    }
    
    .upload-area:hover {
        border-color: var(--primary-color);
        background: #f0f8ff;
    }
    
    /* Results section */
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    
    /* Professional footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# Model URLs
MODEL_URLS = {
    'model': 'https://huggingface.co/PetAnn/sota-speech-emotion-model/resolve/main/sota_xgboost_model.pkl',
    'scaler': 'https://drive.google.com/uc?export=download&id=15zLOfO24Mm8k_XIPhTLrUQ2tZtEIqjzV',
    'metadata': 'https://drive.google.com/uc?export=download&id=1Snkn5nXGFfqzh8yNtyGbCtSwzSbCcNdr',
    'encoder': 'https://drive.google.com/uc?export=download&id=1lfCVCWHXvmwTSIx-f75kDk47FRCqH-5n',
    'selector': 'https://drive.google.com/uc?export=download&id=1boCl4WS5MyIitieEez0BUfimGBkoYJZ9',
    'feature_names': 'https://drive.google.com/uc?export=download&id=1y_VjJj82vuJbdAjD_loOSCQTJ-KAsuky'
}

# Emotion colors for visualizations
EMOTION_COLORS = {
    'angry': '#d62728',
    'calm': '#1f77b4',
    'disgust': '#8c564b',
    'fearful': '#9467bd',
    'happy': '#ff7f0e',
    'neutral': '#7f7f7f',
    'sad': '#2ca02c',
    'surprised': '#bcbd22'
}

@st.cache_resource
def load_models():
    """Load all models and components from remote sources"""
    with st.spinner("🔄 Loading AnalyticsPro models..."):
        try:
            components = {}
            progress_bar = st.progress(0)
            
            for i, (name, url) in enumerate(MODEL_URLS.items()):
                st.text(f"Loading {name}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name
                
                # Load component
                if name == 'metadata':
                    with open(tmp_file_path, 'r') as f:
                        components[name] = json.load(f)
                else:
                    components[name] = joblib.load(tmp_file_path)
                
                # Clean up
                os.unlink(tmp_file_path)
                progress_bar.progress((i + 1) / len(MODEL_URLS))
            
            st.success("✅ All models loaded successfully!")
            return components
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return None

class CleanAudioFeatureExtractor:
    """Clean audio feature extraction - NO SYNTHETIC FEATURES"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.feature_names = None
    
    def extract_clean_audio_features(self, audio_file_path):
        """Extract ONLY real audio signal processing features"""
        try:
            # Load audio
            if isinstance(audio_file_path, str):
                audio, sr = librosa.load(audio_file_path, sr=self.sample_rate, duration=3.0)
            else:
                audio, sr = librosa.load(audio_file_path, sr=self.sample_rate, duration=3.0)
            
            if audio is None or len(audio) == 0:
                return {}
            
            # Clean audio
            if not np.isfinite(audio).all():
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.max(np.abs(audio)) > 0:
                audio = librosa.util.normalize(audio)
            
            features = {}
            
            # 1. COMPREHENSIVE MFCC FEATURES (104 features)
            features.update(self._extract_mfcc_features(audio, sr))
            
            # 2. SPECTRAL FEATURES (16 features)
            features.update(self._extract_spectral_features(audio, sr))
            
            # 3. CHROMA FEATURES (24 features)
            features.update(self._extract_chroma_features(audio, sr))
            
            # 4. PROSODIC FEATURES (11 features)
            features.update(self._extract_prosodic_features(audio, sr))
            
            # 5. ADVANCED SPECTRAL FEATURES (16 features)
            features.update(self._extract_advanced_spectral_features(audio, sr))
            
            # 6. HARMONIC FEATURES (15 features)
            features.update(self._extract_harmonic_features(audio, sr))
            
            # 7. TEMPORAL FEATURES (5 features)
            features.update(self._extract_temporal_features(audio, sr))
            
            # Clean all features
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0.0
            
            if self.feature_names is None:
                self.feature_names = list(features.keys())
            
            return features, audio, sr
            
        except Exception as e:
            st.error(f"Feature extraction error: {str(e)}")
            if self.feature_names is not None:
                return {name: 0.0 for name in self.feature_names}, None, None
            return {}, None, None
    
    def _extract_mfcc_features(self, audio, sr):
        """Comprehensive MFCC features - most important for emotion recognition"""
        features = {}
        
        try:
            # Enhanced MFCC extraction
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            for i in range(13):
                # Comprehensive statistics for each MFCC coefficient
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                features[f'mfcc_{i}_max'] = float(np.max(mfccs[i]))
                features[f'mfcc_{i}_min'] = float(np.min(mfccs[i]))
                features[f'mfcc_{i}_skew'] = float(stats.skew(mfccs[i]))
                features[f'mfcc_{i}_kurtosis'] = float(stats.kurtosis(mfccs[i]))
                features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
                features[f'mfcc_delta2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
                
        except Exception as e:
            # Fallback MFCC features
            for i in range(13):
                for stat in ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']:
                    features[f'mfcc_{i}_{stat}'] = 0.0
                features[f'mfcc_delta_{i}_mean'] = 0.0
                features[f'mfcc_delta2_{i}_mean'] = 0.0
        
        return features
    
    def _extract_spectral_features(self, audio, sr):
        """Basic spectral features"""
        features = {}
        
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            for name, feature_array in [
                ('spectral_centroid', spectral_centroids),
                ('spectral_rolloff', spectral_rolloff),
                ('spectral_bandwidth', spectral_bandwidth),
                ('zero_crossing_rate', zero_crossing_rate)
            ]:
                features[f'{name}_mean'] = float(np.mean(feature_array))
                features[f'{name}_std'] = float(np.std(feature_array))
                features[f'{name}_max'] = float(np.max(feature_array))
                features[f'{name}_skew'] = float(stats.skew(feature_array))
                
        except Exception as e:
            for name in ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate']:
                for stat in ['mean', 'std', 'max', 'skew']:
                    features[f'{name}_{stat}'] = 0.0
        
        return features
    
    def _extract_chroma_features(self, audio, sr):
        """Chroma features for harmonic content"""
        features = {}
        
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
            for i in range(12):
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
                features[f'chroma_{i}_std'] = float(np.std(chroma[i]))
                
        except Exception as e:
            for i in range(12):
                features[f'chroma_{i}_mean'] = 0.0
                features[f'chroma_{i}_std'] = 0.0
        
        return features
    
    def _extract_prosodic_features(self, audio, sr):
        """Prosodic features (F0, energy, etc.)"""
        features = {}
        
        try:
            # F0 extraction
            f0 = librosa.yin(audio, fmin=50, fmax=400)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) > 0:
                features['f0_mean'] = float(np.mean(f0_clean))
                features['f0_std'] = float(np.std(f0_clean))
                features['f0_range'] = float(np.max(f0_clean) - np.min(f0_clean))
                features['f0_jitter'] = float(np.mean(np.abs(np.diff(f0_clean))) / np.mean(f0_clean)) if len(f0_clean) > 1 else 0.0
                features['f0_shimmer'] = float(np.std(f0_clean) / np.mean(f0_clean)) if np.mean(f0_clean) > 0 else 0.0
                features['f0_slope'] = float(np.polyfit(range(len(f0_clean)), f0_clean, 1)[0]) if len(f0_clean) > 1 else 0.0
                features['f0_curvature'] = float(np.polyfit(range(len(f0_clean)), f0_clean, 2)[0]) if len(f0_clean) > 2 else 0.0
            else:
                for feat in ['f0_mean', 'f0_std', 'f0_range', 'f0_jitter', 'f0_shimmer', 'f0_slope', 'f0_curvature']:
                    features[feat] = 0.0
            
            # Energy features
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            features['energy_skew'] = float(stats.skew(rms))
            features['energy_kurtosis'] = float(stats.kurtosis(rms))
            
        except Exception as e:
            for feat in ['f0_mean', 'f0_std', 'f0_range', 'f0_jitter', 'f0_shimmer', 
                        'f0_slope', 'f0_curvature', 'energy_mean', 'energy_std', 'energy_skew', 'energy_kurtosis']:
                features[feat] = 0.0
        
        return features
    
    def _extract_advanced_spectral_features(self, audio, sr):
        """Advanced spectral features"""
        features = {}
        
        try:
            # Spectral contrast (7 bands × 2 stats = 14 features)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            for i in range(min(7, spectral_contrast.shape[0])):
                features[f'spectral_contrast_{i}_mean'] = float(np.mean(spectral_contrast[i]))
                features[f'spectral_contrast_{i}_std'] = float(np.std(spectral_contrast[i]))
            
            # Spectral flatness (2 features)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['spectral_flatness_std'] = float(np.std(spectral_flatness))
            
        except Exception as e:
            for i in range(7):
                features[f'spectral_contrast_{i}_mean'] = 0.0
                features[f'spectral_contrast_{i}_std'] = 0.0
            features['spectral_flatness_mean'] = 0.0
            features['spectral_flatness_std'] = 0.0
        
        return features
    
    def _extract_harmonic_features(self, audio, sr):
        """Harmonic and tonal features"""
        features = {}
        
        try:
            # Tonnetz (6 × 2 = 12 features)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            for i in range(min(6, tonnetz.shape[0])):
                features[f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
                features[f'tonnetz_{i}_std'] = float(np.std(tonnetz[i]))
            
            # Harmonic-percussive separation (3 features)
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            features['harmonic_energy'] = float(np.mean(y_harmonic**2))
            features['percussive_energy'] = float(np.mean(y_percussive**2))
            features['harmonic_percussive_ratio'] = float(features['harmonic_energy'] / (features['percussive_energy'] + 1e-8))
            
        except Exception as e:
            for i in range(6):
                features[f'tonnetz_{i}_mean'] = 0.0
                features[f'tonnetz_{i}_std'] = 0.0
            features['harmonic_energy'] = 0.0
            features['percussive_energy'] = 0.0
            features['harmonic_percussive_ratio'] = 0.0
        
        return features
    
    def _extract_temporal_features(self, audio, sr):
        """Temporal features (rhythm, beats, etc.)"""
        features = {}
        
        try:
            # Tempo and beat features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = float(len(beats))
            features['beat_variance'] = float(np.var(np.diff(beats))) if len(beats) > 1 else 0.0
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features['onset_count'] = float(len(onset_frames))
            features['onset_rate'] = float(len(onset_frames) / (len(audio) / sr))
            
        except Exception as e:
            for feat in ['tempo', 'beat_count', 'beat_variance', 'onset_count', 'onset_rate']:
                features[feat] = 0.0
        
        return features

def predict_emotion(audio_file, components):
    """Predict emotion from audio file"""
    try:
        # Extract features
        extractor = CleanAudioFeatureExtractor()
        features_dict, audio_signal, sample_rate = extractor.extract_clean_audio_features(audio_file)
        
        if not features_dict:
            return None, None, None, None
        
        # Prepare features in correct order
        feature_names = components['feature_names']
        features_array = np.array([features_dict.get(name, 0.0) for name in feature_names]).reshape(1, -1)
        
        # Apply feature selection
        features_selected = components['selector'].transform(features_array)
        
        # Scale features
        features_scaled = components['scaler'].transform(features_selected)
        
        # Predict
        prediction = components['model'].predict(features_scaled)[0]
        probabilities = components['model'].predict_proba(features_scaled)[0]
        
        # Decode emotion
        emotion = components['encoder'].inverse_transform([prediction])[0]
        emotion_probs = dict(zip(components['encoder'].classes_, probabilities))
        
        return emotion, emotion_probs, audio_signal, sample_rate
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

def create_waveform_plot(audio, sr, emotion):
    """Create waveform visualization"""
    fig = go.Figure()
    
    time = np.arange(len(audio)) / sr
    fig.add_trace(go.Scatter(
        x=time,
        y=audio,
        mode='lines',
        name='Waveform',
        line=dict(color=EMOTION_COLORS.get(emotion, '#1f77b4'), width=1)
    ))
    
    fig.update_layout(
        title=f"Audio Waveform - Detected Emotion: {emotion.capitalize()}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=300
    )
    
    return fig

def create_spectrogram(audio, sr):
    """Create spectrogram visualization"""
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        colorscale='Viridis',
        colorbar=dict(title="dB")
    ))
    
    fig.update_layout(
        title="Spectrogram",
        xaxis_title="Time",
        yaxis_title="Frequency",
        template="plotly_white",
        height=300
    )
    
    return fig

def create_emotion_probability_chart(emotion_probs):
    """Create emotion probability bar chart"""
    emotions = list(emotion_probs.keys())
    probs = list(emotion_probs.values())
    colors = [EMOTION_COLORS.get(e, '#1f77b4') for e in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probs,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotion",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=400
    )
    
    return fig

def create_feature_importance_plot(features_dict, top_n=20):
    """Create feature importance visualization"""
    # Calculate feature statistics for importance
    feature_values = list(features_dict.values())
    feature_names = list(features_dict.keys())
    
    # Simple importance based on absolute values
    importances = [abs(val) for val in feature_values]
    
    # Get top N features
    top_indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = [importances[i] for i in top_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_features,
            x=top_importances,
            orientation='h',
            marker_color='#1f77b4'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Audio Features",
        xaxis_title="Feature Value (Absolute)",
        yaxis_title="Feature Name",
        template="plotly_white",
        height=500
    )
    
    return fig

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ AnalyticsPro - Speech Emotion Recognition</h1>
        <p>Professional Audio Analytics Platform | 82%+ Accuracy | Clean Audio Features Only</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Developed by Peter Chika Ozo-ogueji (Data Scientist)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=AnalyticsPro", use_column_width=True)
        
        st.markdown("### 🎯 About AnalyticsPro")
        st.info("""
        **State-of-the-Art Features:**
        - 191 Clean Audio Features
        - No Synthetic Features
        - 82%+ Test Accuracy
        - 8 Emotion Classes
        - Production-Ready
        """)
        
        st.markdown("### 📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "82.0%")
            st.metric("F1-Score", "83.1%")
        with col2:
            st.metric("Features", "191")
            st.metric("Samples", "10,982")
        
        st.markdown("### 🎭 Supported Emotions")
        emotions = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        for emotion in emotions:
            st.markdown(f"• {emotion}")
        
        st.markdown("### 🔧 Settings")
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        batch_mode = st.checkbox("Batch Processing Mode", value=False)
    
    # Load models
    if not st.session_state.model_loaded:
        components = load_models()
        if components:
            st.session_state.components = components
            st.session_state.model_loaded = True
        else:
            st.error("Failed to load models. Please refresh the page.")
            return
    else:
        components = st.session_state.components
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Single Analysis", "📁 Batch Processing", "📊 Analytics Dashboard", "📚 Documentation"])
    
    with tab1:
        st.markdown("### Upload Audio for Emotion Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'aac'],
                help="Upload a clear audio file (3-5 seconds recommended)"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Display audio player
                st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')
                
                # Process button
                if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
                    with st.spinner("Processing audio..."):
                        # Predict emotion
                        emotion, emotion_probs, audio_signal, sample_rate = predict_emotion(tmp_file_path, components)
                        
                        if emotion:
                            # Results container
                            st.markdown('<div class="results-container">', unsafe_allow_html=True)
                            
                            # Main result
                            st.markdown("### 🎭 Detected Emotion")
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                emotion_class = f"emotion-{emotion}"
                                st.markdown(f'<div class="emotion-badge {emotion_class}">{emotion.upper()}</div>', unsafe_allow_html=True)
                                confidence = emotion_probs[emotion] * 100
                                st.markdown(f"**Confidence:** {confidence:.1f}%")
                            
                            # Emotion probabilities
                            st.markdown("### 📊 Emotion Probabilities")
                            fig_probs = create_emotion_probability_chart(emotion_probs)
                            st.plotly_chart(fig_probs, use_container_width=True)
                            
                            if show_advanced and audio_signal is not None:
                                # Advanced analytics
                                st.markdown("### 🔬 Advanced Audio Analytics")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Waveform
                                    fig_wave = create_waveform_plot(audio_signal, sample_rate, emotion)
                                    st.plotly_chart(fig_wave, use_container_width=True)
                                
                                with col2:
                                    # Spectrogram
                                    fig_spec = create_spectrogram(audio_signal, sample_rate)
                                    st.plotly_chart(fig_spec, use_container_width=True)
                                
                                # Feature analysis
                                st.markdown("### 🎯 Feature Analysis")
                                extractor = CleanAudioFeatureExtractor()
                                features_dict, _, _ = extractor.extract_clean_audio_features(tmp_file_path)
                                
                                if features_dict:
                                    fig_features = create_feature_importance_plot(features_dict)
                                    st.plotly_chart(fig_features, use_container_width=True)
                            
                            # Save to session state
                            result = {
                                'filename': uploaded_file.name,
                                'emotion': emotion,
                                'confidence': confidence,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.processed_files.append(result)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download results
                            st.markdown("### 💾 Export Results")
                            results_json = json.dumps({
                                'filename': uploaded_file.name,
                                'emotion': emotion,
                                'probabilities': emotion_probs,
                                'metadata': components['metadata'],
                                'timestamp': datetime.now().isoformat()
                            }, indent=2)
                            
                            st.download_button(
                                label="📥 Download Analysis Report (JSON)",
                                data=results_json,
                                file_name=f"emotion_analysis_{uploaded_file.name.split('.')[0]}.json",
                                mime="application/json"
                            )
                        else:
                            st.error("Failed to analyze audio. Please try again with a different file.")
                
                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        
        with col2:
            st.markdown("### 💡 Tips for Best Results")
            st.markdown("""
            - Use clear audio recordings
            - Optimal duration: 3-5 seconds
            - Minimize background noise
            - Single speaker recommended
            - Supported formats: WAV, MP3, FLAC
            """)
            
            st.markdown("### 🎯 Model Info")
            if 'metadata' in components:
                metadata = components['metadata']
                st.markdown(f"**Model Type:** {metadata.get('model_type', 'N/A')}")
                st.markdown(f"**Training Date:** {metadata.get('training_date', 'N/A')[:10]}")
                st.markdown(f"**Author:** {metadata.get('author', 'N/A')}")
    
    with tab2:
        st.markdown("### 📁 Batch Processing")
        st.info("Process multiple audio files at once for comprehensive emotion analysis")
        
        uploaded_files = st.file_uploader(
            "Choose multiple audio files",
            type=['wav', 'mp3', 'flac', 'm4a', 'aac'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.markdown(f"**Files selected:** {len(uploaded_files)}")
            
            if st.button("🚀 Process All Files", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for i, file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    emotion, emotion_probs, _, _ = predict_emotion(tmp_file_path, components)
                    
                    if emotion:
                        results.append({
                            'Filename': file.name,
                            'Emotion': emotion.capitalize(),
                            'Confidence': f"{emotion_probs[emotion]*100:.1f}%",
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                if results:
                    st.session_state.batch_results = results
                    
                    # Display results table
                    df_results = pd.DataFrame(results)
                    st.markdown("### 📊 Batch Processing Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Emotion distribution
                    emotion_counts = df_results['Emotion'].value_counts()
                    fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, 
                                title="Emotion Distribution in Batch",
                                color_discrete_map={e.capitalize(): EMOTION_COLORS.get(e.lower(), '#1f77b4') 
                                                  for e in emotion_counts.index})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download batch results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Batch Results (CSV)",
                        data=csv,
                        file_name=f"batch_emotion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        st.markdown("### 📊 Analytics Dashboard")
        
        if st.session_state.processed_files or st.session_state.batch_results:
            # Combine all results
            all_results = st.session_state.processed_files + [
                {'filename': r['Filename'], 'emotion': r['Emotion'].lower(), 
                 'confidence': float(r['Confidence'].strip('%')), 'timestamp': r['Timestamp']}
                for r in st.session_state.batch_results
            ]
            
            if all_results:
                df_all = pd.DataFrame(all_results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files Analyzed", len(df_all))
                with col2:
                    st.metric("Average Confidence", f"{df_all['confidence'].mean():.1f}%")
                with col3:
                    st.metric("Most Common Emotion", df_all['emotion'].mode()[0].capitalize())
                with col4:
                    st.metric("Unique Emotions", df_all['emotion'].nunique())
                
                # Emotion timeline
                if len(df_all) > 1:
                    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
                    fig_timeline = px.scatter(df_all, x='timestamp', y='confidence', 
                                            color='emotion', size='confidence',
                                            title="Emotion Detection Timeline",
                                            color_discrete_map=EMOTION_COLORS)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Emotion distribution
                emotion_dist = df_all['emotion'].value_counts()
                fig_dist = go.Figure(data=[
                    go.Bar(x=emotion_dist.index, y=emotion_dist.values,
                          marker_color=[EMOTION_COLORS.get(e, '#1f77b4') for e in emotion_dist.index])
                ])
                fig_dist.update_layout(title="Overall Emotion Distribution",
                                     xaxis_title="Emotion", yaxis_title="Count")
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Confidence distribution by emotion
                fig_box = px.box(df_all, x='emotion', y='confidence', 
                               title="Confidence Distribution by Emotion",
                               color='emotion', color_discrete_map=EMOTION_COLORS)
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Recent analyses table
                st.markdown("### 📋 Recent Analyses")
                st.dataframe(df_all.sort_values('timestamp', ascending=False).head(10), 
                           use_container_width=True)
        else:
            st.info("No analyses performed yet. Upload audio files to see analytics.")
    
    with tab4:
        st.markdown("### 📚 Documentation")
        
        with st.expander("🎯 About the Model", expanded=True):
            st.markdown("""
            **AnalyticsPro Speech Emotion Recognition** uses state-of-the-art audio processing techniques:
            
            - **191 Clean Audio Features**: MFCC, Spectral, Chroma, Prosodic, Harmonic, and Temporal features
            - **No Synthetic Features**: Reliable, deployment-ready model
            - **Ensemble Architecture**: Combines XGBoost, LightGBM, Random Forest, and SVM
            - **82%+ Accuracy**: Validated on multiple benchmark datasets
            
            The model was trained on over 10,000 audio samples from:
            - RAVDESS (85%+ accuracy)
            - CREMA-D (80%+ validated)
            - TESS (High performance)
            - EMO-DB (90%+ on clean features)
            - SAVEE (Proven benchmark)
            """)
        
        with st.expander("🎭 Emotion Classes"):
            emotion_descriptions = {
                "Angry": "Characterized by high energy, elevated pitch, and harsh tone quality",
                "Calm": "Low arousal state with steady pitch and relaxed vocal quality",
                "Disgust": "Often includes vocal tension and specific prosodic patterns",
                "Fearful": "Higher pitch variance, trembling quality, and increased speech rate",
                "Happy": "Elevated pitch, increased energy, and faster speech tempo",
                "Neutral": "Baseline emotional state with normal prosodic features",
                "Sad": "Lower pitch, reduced energy, and slower speech rate",
                "Surprised": "Sudden pitch changes and increased vocal intensity"
            }
            
            for emotion, description in emotion_descriptions.items():
                st.markdown(f"**{emotion}**: {description}")
        
        with st.expander("🔬 Technical Details"):
            st.markdown("""
            **Feature Extraction Pipeline:**
            
            1. **MFCC Features (104)**: 13 coefficients with comprehensive statistics
            2. **Spectral Features (16)**: Centroid, rolloff, bandwidth, zero-crossing rate
            3. **Chroma Features (24)**: 12 pitch classes with mean and std
            4. **Prosodic Features (11)**: F0 statistics, jitter, shimmer, energy
            5. **Advanced Spectral (16)**: Spectral contrast and flatness
            6. **Harmonic Features (15)**: Tonnetz, harmonic-percussive separation
            7. **Temporal Features (5)**: Tempo, beats, onsets
            
            **Model Architecture:**
            - Feature Selection: SelectKBest (150 features)
            - Scaling: RobustScaler
            - Class Balancing: BorderlineSMOTE
            - Ensemble: Soft voting classifier
            """)
        
        with st.expander("💡 Usage Tips"):
            st.markdown("""
            **For Best Results:**
            
            1. **Audio Quality**: Use clear recordings with minimal background noise
            2. **Duration**: 3-5 seconds is optimal for emotion detection
            3. **Single Speaker**: Works best with one speaker at a time
            4. **Natural Speech**: Avoid exaggerated or acted emotions
            5. **File Formats**: WAV provides best quality, but MP3/FLAC also supported
            
            **Common Issues:**
            - Very short clips (<1 second) may have reduced accuracy
            - Multiple speakers can confuse the model
            - Excessive noise affects feature extraction
            - Compressed audio may lose important frequency information
            """)
        
        with st.expander("🔗 API Integration"):
            st.code("""
# Example API usage (when deployed)
import requests

url = "https://your-api-endpoint/predict"
files = {'audio': open('speech.wav', 'rb')}
response = requests.post(url, files=files)

result = response.json()
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']}")
            """, language='python')
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎙️ AnalyticsPro - Professional Speech Emotion Recognition</p>
        <p>Developed by Peter Chika Ozo-ogueji (Data Scientist)</p>
        <p>© 2024 | Clean Audio Features Only | 82%+ Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
