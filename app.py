import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import requests
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import joblib
import json
import re
from scipy import stats

# Model URLs
MODEL_URLS = {
    'model': 'https://drive.google.com/uc?export=download&id=1kVZ6qg0a_8DNu1yn_hGYlc7mTQokk1CS&confirm=t',
    'scaler': 'https://drive.google.com/uc?export=download&id=1kjrAEPwVLbKyztSYxGmapysi1_prkJEK&confirm=t',
    'metadata': 'https://drive.google.com/uc?export=download&id=1TLlvM3SSIrUnz-0isLQGh5fUtV-4CzPu&confirm=t',
    'label_encoder': 'https://drive.google.com/uc?export=download&id=1o-PX_oquJCmzyrsgq1Oh1Hvv4uUfRZ5n&confirm=t',
    'feature_selector': 'https://drive.google.com/uc?export=download&id=1u4tX6Xzd9LOJ12PkYKYT3rWnHzncOngi&confirm=t',
    'feature_names': 'https://drive.google.com/uc?export=download&id=1lwFLlbCrFPLvfiPvxkFNTWyw91djgzyK&confirm=t'
}

st.set_page_config(
    page_title="🔧 Advanced Feature Reconstruction",
    page_icon="🔧",
    layout="wide"
)

@st.cache_data
def load_models_final():
    """Load models with proper error handling"""
    models = {}
    success_count = 0
    
    with st.spinner("Loading all model components..."):
        progress_bar = st.progress(0)
        
        for i, (key, url) in enumerate(MODEL_URLS.items()):
            progress_bar.progress(i / len(MODEL_URLS))
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=300)
                response.raise_for_status()
                content = response.content
                
                if key == 'metadata':
                    models[key] = json.loads(content.decode('utf-8'))
                else:
                    models[key] = joblib.load(io.BytesIO(content))
                
                success_count += 1
                
            except Exception as e:
                st.error(f"Failed to load {key}: {e}")
                models[key] = None
        
        progress_bar.progress(1.0)
    
    if success_count >= 5:
        st.success(f"✅ Loaded {success_count}/6 model components!")
    else:
        st.error(f"❌ Only {success_count}/6 components loaded")
        return None
    
    return models

def extract_comprehensive_features(audio_file, feature_names, sample_rate=22050):
    """Extract comprehensive features matching training patterns"""
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=sample_rate, duration=3.0)
        
        if audio is None or len(audio) == 0:
            st.error("Failed to load audio")
            return None
        
        # Normalize audio
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)
        
        features = {}
        
        # 1. MFCC Features (comprehensive extraction)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                features[f'mfcc_{i}_max'] = float(np.max(mfccs[i]))
                features[f'mfcc_{i}_min'] = float(np.min(mfccs[i]))
                features[f'mfcc_{i}_skew'] = float(stats.skew(mfccs[i]))
                features[f'mfcc_{i}_kurtosis'] = float(stats.kurtosis(mfccs[i]))
                features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
                features[f'mfcc_delta2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
        except Exception as e:
            st.warning(f"MFCC extraction failed: {e}")
        
        # 2. Spectral Features
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
            st.warning(f"Spectral extraction failed: {e}")
        
        # 3. Chroma Features
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
            for i in range(12):
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
                features[f'chroma_{i}_std'] = float(np.std(chroma[i]))
        except Exception as e:
            st.warning(f"Chroma extraction failed: {e}")
        
        # 4. Prosodic Features
        try:
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
            
            # Energy features
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            features['energy_skew'] = float(stats.skew(rms))
            features['energy_kurtosis'] = float(stats.kurtosis(rms))
        except Exception as e:
            st.warning(f"Prosodic extraction failed: {e}")
        
        # 5. Advanced spectral features
        try:
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            for i in range(min(7, spectral_contrast.shape[0])):
                features[f'spectral_contrast_{i}_mean'] = float(np.mean(spectral_contrast[i]))
                features[f'spectral_contrast_{i}_std'] = float(np.std(spectral_contrast[i]))
            
            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['spectral_flatness_std'] = float(np.std(spectral_flatness))
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            for i in range(min(6, tonnetz.shape[0])):
                features[f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
                features[f'tonnetz_{i}_std'] = float(np.std(tonnetz[i]))
        except Exception as e:
            st.warning(f"Advanced spectral extraction failed: {e}")
        
        # 6. Additional features to reach ~155 real features
        try:
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = float(len(beats))
            features['beat_variance'] = float(np.var(np.diff(beats))) if len(beats) > 1 else 0.0
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features['onset_count'] = float(len(onset_frames))
            features['onset_rate'] = float(len(onset_frames) / (len(audio) / sr))
            
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            features['harmonic_energy'] = float(np.mean(y_harmonic**2))
            features['percussive_energy'] = float(np.mean(y_percussive**2))
            features['harmonic_percussive_ratio'] = float(features['harmonic_energy'] / (features['percussive_energy'] + 1e-8))
        except Exception as e:
            st.warning(f"Additional feature extraction failed: {e}")
        
        # 7. ADVANCED SYNTHETIC FEATURE RECONSTRUCTION
        # Instead of using simple defaults, create realistic synthetic features
        # based on the real audio features
        
        real_feature_count = len(features)
        st.info(f"🎵 Extracted {real_feature_count} real audio features")
        
        # Create feature array with sophisticated synthetic feature generation
        feature_values = []
        missing_count = 0
        
        # Get statistics from real features for generating synthetic ones
        real_values = list(features.values())
        real_mean = np.mean(real_values)
        real_std = np.std(real_values)
        real_range = np.max(real_values) - np.min(real_values)
        
        for i, name in enumerate(feature_names):
            if name in features:
                value = features[name]
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_values.append(float(value))
            else:
                missing_count += 1
                
                # SOPHISTICATED SYNTHETIC FEATURE GENERATION
                if 'vit_feature' in name:
                    # Vision Transformer features - create correlated patterns
                    vit_index = int(name.split('_')[-1]) if name.split('_')[-1].isdigit() else 0
                    
                    # Create features correlated with audio energy and spectral content
                    energy_proxy = features.get('energy_mean', 0.1)
                    spectral_proxy = features.get('spectral_centroid_mean', 2000) / 2000
                    
                    # Different patterns for different VIT feature indices
                    if vit_index < 10:
                        # Early VIT features - correlate with energy
                        value = energy_proxy * np.random.normal(0.1, 0.02)
                    elif vit_index < 30:
                        # Middle VIT features - correlate with spectral content
                        value = spectral_proxy * np.random.normal(0.05, 0.01)
                    else:
                        # Late VIT features - correlate with pitch
                        f0_proxy = features.get('f0_mean', 150) / 150
                        value = f0_proxy * np.random.normal(0.02, 0.005)
                
                elif 'graph' in name:
                    # Graph features - create realistic graph-like values
                    if 'nodes' in name:
                        # Node count - should be reasonable integer-like
                        value = np.random.uniform(10, 50)
                    elif 'edges' in name:
                        # Edge count - should be related to nodes
                        value = np.random.uniform(20, 100)
                    elif 'density' in name:
                        # Graph density - between 0 and 1
                        value = np.random.uniform(0.1, 0.8)
                    elif 'clustering' in name:
                        # Clustering coefficient - between 0 and 1
                        value = np.random.uniform(0.2, 0.9)
                    elif 'degree' in name:
                        if 'avg' in name:
                            value = np.random.uniform(2, 8)
                        elif 'std' in name:
                            value = np.random.uniform(1, 3)
                        else:
                            value = np.random.uniform(1, 10)
                    else:
                        value = np.random.uniform(0, 1)
                
                elif 'quantum' in name:
                    # Quantum features - very small, specialized values
                    if 'entanglement' in name:
                        # Entanglement should be small positive
                        if 'mean' in name:
                            value = np.random.uniform(0.001, 0.01)
                        else:  # std
                            value = np.random.uniform(0.0001, 0.001)
                    elif 'coherence' in name:
                        # Coherence between 0 and 1, but small
                        value = np.random.uniform(0.01, 0.1)
                    else:
                        value = np.random.uniform(0, 0.001)
                
                else:
                    # Unknown synthetic features - use audio-correlated values
                    correlation_factor = np.random.choice([
                        features.get('energy_mean', 0.1),
                        features.get('spectral_centroid_mean', 2000) / 2000,
                        features.get('f0_mean', 150) / 150
                    ])
                    value = correlation_factor * np.random.normal(0.01, 0.005)
                
                feature_values.append(float(value))
        
        # Create DataFrame with proper names
        feature_df = pd.DataFrame([feature_values], columns=feature_names)
        
        st.success(f"✅ Created feature set: {real_feature_count} real + {missing_count} reconstructed synthetic features")
        
        return feature_df
        
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def predict_with_reconstructed_features(feature_df, models):
    """Make prediction using reconstructed features"""
    
    try:
        if feature_df is None:
            return None, None, None
        
        model = models['model']
        scaler = models['scaler']
        feature_selector = models['feature_selector']
        label_encoder = models['label_encoder']
        
        st.info(f"🔬 Input DataFrame shape: {feature_df.shape}")
        
        X = feature_df.values
        
        # Apply preprocessing
        if feature_selector:
            X = feature_selector.transform(X)
            st.info(f"✅ After feature selection: {X.shape}")
        
        if scaler:
            X = scaler.transform(X)
            st.info(f"✅ After scaling: range [{X.min():.3f}, {X.max():.3f}]")
        
        # Create named DataFrame for LightGBM
        if feature_selector and hasattr(feature_selector, 'get_support'):
            selected_mask = feature_selector.get_support()
            selected_feature_names = feature_df.columns[selected_mask]
            X_named = pd.DataFrame(X, columns=selected_feature_names)
        else:
            X_named = pd.DataFrame(X, columns=feature_df.columns)
        
        # Make prediction
        prediction = model.predict(X_named)[0]
        probabilities = model.predict_proba(X_named)[0]
        
        emotion = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Get all emotion probabilities
        emotion_probs = {}
        for i, prob in enumerate(probabilities):
            emo = label_encoder.inverse_transform([i])[0]
            emotion_probs[emo] = prob
        
        return emotion, confidence, emotion_probs
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

def main():
    st.title("🔧 Advanced Synthetic Feature Reconstruction")
    st.markdown("### 🎯 **Final Solution: Realistic Synthetic Features**")
    
    st.info("🧠 **New Approach**: Instead of simple defaults, generate realistic synthetic features correlated with real audio patterns!")
    
    # Load models
    models = load_models_final()
    
    if not models:
        st.error("❌ Failed to load models")
        return
    
    st.sidebar.success("✅ All Models Loaded!")
    st.sidebar.json({
        "Approach": "Advanced Reconstruction",
        "Strategy": "Audio-Correlated Synthetics",
        "Features": "214 (Smart Generated)",
        "Status": "Testing"
    })
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎵 Test Advanced Feature Reconstruction")
        st.success("🎉 **NEW APPROACH**: Sophisticated synthetic feature generation!")
        
        st.markdown("""
        **How this works:**
        1. **Extract 155 real audio features** normally
        2. **Generate 59 synthetic features** correlated with real audio:
           - VIT features → Correlate with energy & spectral content  
           - Graph features → Realistic graph topology values
           - Quantum features → Physics-appropriate small values
        3. **Create realistic feature distribution** the model might recognize
        """)
        
        uploaded_file = st.file_uploader(
            "Upload your angry customer audio",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Let's see if advanced reconstruction fixes the predictions!"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            with st.spinner('🔬 Generating sophisticated feature reconstruction...'):
                
                feature_names = models['feature_names']
                feature_df = extract_comprehensive_features(uploaded_file, feature_names)
                
                if feature_df is not None:
                    emotion, confidence, emotion_probs = predict_with_reconstructed_features(feature_df, models)
                    
                    if emotion:
                        # Display results
                        st.success(f"🎯 **Predicted Emotion:** {emotion.title()}")
                        st.info(f"🎲 **Confidence:** {confidence:.1%}")
                        
                        # Check if we finally got it right
                        if emotion in ['angry', 'fearful', 'sad'] and confidence > 0.4:
                            st.balloons()
                            st.success("🎉 **BREAKTHROUGH!** Advanced reconstruction appears to work!")
                        elif emotion not in ['disgust', 'happy']:
                            st.success("✅ **PROGRESS!** At least not disgust or happy anymore!")
                        
                        # Visualization
                        st.subheader("📊 Advanced Reconstruction Results")
                        
                        prob_df = pd.DataFrame(
                            list(emotion_probs.items()),
                            columns=['Emotion', 'Probability']
                        ).sort_values('Probability', ascending=True)
                        
                        fig = px.bar(
                            prob_df,
                            x='Probability',
                            y='Emotion',
                            orientation='h',
                            title="Advanced Feature Reconstruction Predictions",
                            color='Probability',
                            color_continuous_scale='RdYlBu_r'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Top predictions
                        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                        st.subheader("🏆 Top 3 Predictions")
                        for i, (emo, prob) in enumerate(sorted_emotions[:3]):
                            icon = "🎯" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.write(f"{icon} {i+1}. **{emo.title()}**: {prob:.1%}")
                        
                        # Analysis
                        st.subheader("📈 Prediction Analysis")
                        if confidence > 0.5:
                            st.success("🎯 Good confidence level!")
                        elif confidence > 0.35:
                            st.info("🤔 Moderate confidence - better than before")
                        else:
                            st.warning("😐 Still low confidence")
                        
                        # Compare with simple strategies
                        st.info("💡 **If this works better**, we can implement it in your main app!")
                        
                    else:
                        st.error("❌ Prediction failed")
                else:
                    st.error("❌ Feature extraction failed")
    
    with col2:
        st.header("🧠 Advanced Strategy")
        
        st.subheader("🔬 What's Different")
        st.markdown("""
        **Previous approaches:**
        - Fill synthetic features with zeros
        - Fill with tiny random values  
        - Use simple statistical defaults
        
        **Advanced approach:**
        - **VIT features**: Correlate with audio energy & spectral content
        - **Graph features**: Generate realistic network topology values
        - **Quantum features**: Use physics-appropriate ranges
        - **Audio correlation**: Synthetic features vary with real audio properties
        """)
        
        st.subheader("🎯 Expected Improvement")
        st.markdown("""
        **Theory**: The model was trained with synthetic features that had specific patterns/correlations. By generating synthetic features that correlate with real audio properties, we create a feature distribution closer to what the model saw during training.
        
        **Goal**: Get accurate predictions that match the actual emotion in the audio.
        """)
        
        st.subheader("📊 Success Metrics")
        st.markdown("""
        ✅ **Success**: Angry audio → "angry" prediction  
        ✅ **Success**: Confidence > 50%  
        ✅ **Success**: Realistic emotion distribution  
        ❌ **Failure**: Still predicting happy/disgust  
        ❌ **Failure**: Very low confidence (<30%)  
        """)

if __name__ == "__main__":
    main()
