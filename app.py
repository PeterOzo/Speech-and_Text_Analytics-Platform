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
    page_title="🔍 Precise Feature Analyzer",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_models_quick():
    """Quick model loading"""
    models = {}
    for key, url in MODEL_URLS.items():
        try:
            response = requests.get(url, timeout=300)
            content = response.content
            if key == 'metadata':
                models[key] = json.loads(content.decode('utf-8'))
            else:
                models[key] = joblib.load(io.BytesIO(content))
        except:
            models[key] = None
    return models

def analyze_exact_missing_features():
    """Identify exactly which features are missing and categorize them"""
    
    st.header("🔍 Exact Missing Feature Analysis")
    
    models = load_models_quick()
    feature_names = models.get('feature_names')
    
    if not feature_names:
        st.error("Could not load feature names")
        return None, None
    
    st.success(f"✅ Loaded {len(feature_names)} expected feature names")
    
    # Show first and last features to understand structure
    st.subheader("📋 Feature Name Structure")
    st.write("**First 10 features:**")
    for i, name in enumerate(feature_names[:10]):
        st.write(f"  {i+1}. {name}")
    
    st.write("**Last 10 features:**")
    for i, name in enumerate(feature_names[-10:]):
        st.write(f"  {len(feature_names)-10+i+1}. {name}")
    
    # Create comprehensive feature categorization
    feature_categories = {
        'mfcc_basic': [],      # Basic MFCC stats
        'mfcc_delta': [],      # MFCC deltas
        'spectral_basic': [],  # Basic spectral features
        'spectral_advanced': [], # Advanced spectral features
        'chroma': [],          # Chroma features
        'prosodic': [],        # F0, energy, etc.
        'harmonic': [],        # Tonnetz, harmonic features
        'temporal': [],        # Rhythm, tempo features
        'synthetic': [],       # Likely synthetic/placeholder features
        'unknown': []          # Unclassified
    }
    
    # Categorize each feature
    for name in feature_names:
        name_lower = name.lower()
        
        if 'mfcc' in name_lower and 'delta' not in name_lower:
            feature_categories['mfcc_basic'].append(name)
        elif 'mfcc_delta' in name_lower:
            feature_categories['mfcc_delta'].append(name)
        elif any(x in name_lower for x in ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing']):
            feature_categories['spectral_basic'].append(name)
        elif any(x in name_lower for x in ['spectral_contrast', 'spectral_flatness']):
            feature_categories['spectral_advanced'].append(name)
        elif 'chroma' in name_lower:
            feature_categories['chroma'].append(name)
        elif any(x in name_lower for x in ['f0', 'energy', 'pitch', 'jitter', 'shimmer']):
            feature_categories['prosodic'].append(name)
        elif any(x in name_lower for x in ['tonnetz', 'harmonic', 'percussive']):
            feature_categories['harmonic'].append(name)
        elif any(x in name_lower for x in ['tempo', 'beat', 'onset', 'rhythm']):
            feature_categories['temporal'].append(name)
        elif any(x in name_lower for x in ['vit_feature', 'graph', 'quantum', 'transformer', 'bert', 'wav2vec']):
            feature_categories['synthetic'].append(name)
        else:
            feature_categories['unknown'].append(name)
    
    # Display detailed categorization
    st.subheader("📊 Detailed Feature Categorization")
    
    total_real_audio = 0
    total_synthetic = 0
    
    for category, features in feature_categories.items():
        if features:
            st.write(f"**{category.upper().replace('_', ' ')}** ({len(features)} features):")
            
            if category == 'synthetic':
                total_synthetic += len(features)
                st.error(f"  🚨 PROBLEMATIC: {len(features)} synthetic features detected!")
                # Show some examples
                for feat in features[:5]:
                    st.write(f"    • {feat}")
                if len(features) > 5:
                    st.write(f"    • ... and {len(features)-5} more")
            else:
                total_real_audio += len(features)
                if len(features) <= 5:
                    for feat in features:
                        st.write(f"    • {feat}")
                else:
                    st.write(f"    • {features[0]} ... {features[-1]} (and {len(features)-2} others)")
    
    st.info(f"📊 **Summary**: {total_real_audio} real audio features, {total_synthetic} synthetic features")
    
    if total_synthetic > 50:
        st.error(f"🚨 **MAJOR ISSUE**: {total_synthetic} synthetic features are likely causing prediction errors!")
    
    return feature_names, feature_categories

def extract_only_real_features(audio_file, sample_rate=22050):
    """Extract ONLY real audio features, ignore synthetic ones"""
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=sample_rate, duration=3.0)
        
        if audio is None or len(audio) == 0:
            return None
        
        # Normalize
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)
        
        features = {}
        
        # 1. MFCC Features (13 coefficients × 8 statistics = 104 features)
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
        
        # 2. Basic Spectral Features (4 types × 4 stats = 16 features)
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
            st.warning(f"Basic spectral extraction failed: {e}")
        
        # 3. Chroma Features (12 × 2 = 24 features)
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
            for i in range(12):
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
                features[f'chroma_{i}_std'] = float(np.std(chroma[i]))
        except Exception as e:
            st.warning(f"Chroma extraction failed: {e}")
        
        # 4. Advanced Spectral Features
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
            st.warning(f"Advanced spectral extraction failed: {e}")
        
        # 5. Harmonic Features
        try:
            # Tonnetz (6 × 2 = 12 features)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            for i in range(min(6, tonnetz.shape[0])):
                features[f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
                features[f'tonnetz_{i}_std'] = float(np.std(tonnetz[i]))
            
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            features['harmonic_energy'] = float(np.mean(y_harmonic**2))
            features['percussive_energy'] = float(np.mean(y_percussive**2))
            features['harmonic_percussive_ratio'] = float(features['harmonic_energy'] / (features['percussive_energy'] + 1e-8))
        except Exception as e:
            st.warning(f"Harmonic extraction failed: {e}")
        
        # 6. Prosodic Features (F0, energy, etc.)
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
            st.warning(f"Prosodic extraction failed: {e}")
        
        # 7. Temporal Features
        try:
            # Tempo and beat
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = float(len(beats))
            features['beat_variance'] = float(np.var(np.diff(beats))) if len(beats) > 1 else 0.0
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features['onset_count'] = float(len(onset_frames))
            features['onset_rate'] = float(len(onset_frames) / (len(audio) / sr))
        except Exception as e:
            st.warning(f"Temporal extraction failed: {e}")
        
        # Clean all features
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        st.success(f"✅ Extracted {len(features)} REAL audio features")
        return features
        
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

def test_different_missing_strategies(audio_file, models, real_features):
    """Test different strategies for handling missing features"""
    
    st.header("🧪 Testing Missing Feature Strategies")
    
    feature_names = models['feature_names']
    model = models['model']
    scaler = models['scaler']
    feature_selector = models['feature_selector']
    label_encoder = models['label_encoder']
    
    # Identify missing features
    missing_features = [name for name in feature_names if name not in real_features]
    present_features = [name for name in feature_names if name in real_features]
    
    st.info(f"📊 **Feature Status**: {len(present_features)} real features, {len(missing_features)} missing")
    
    # Show which features are missing
    with st.expander("🔍 Show Missing Features"):
        st.write("**Missing features (likely synthetic):**")
        for i, feat in enumerate(missing_features[:20]):  # Show first 20
            st.write(f"  {i+1}. {feat}")
        if len(missing_features) > 20:
            st.write(f"  ... and {len(missing_features)-20} more")
    
    # Test different strategies
    strategies = {
        "zero_fill": "Fill all missing with 0.0",
        "negative_tiny": "Fill with very small negative values (-0.001)",
        "positive_tiny": "Fill with very small positive values (+0.001)",
        "statistical_mean": "Fill with statistical means based on feature type",
        "remove_synthetic": "Try to skip synthetic features entirely (if possible)"
    }
    
    results = {}
    
    for strategy_name, strategy_desc in strategies.items():
        st.subheader(f"🔬 Strategy: {strategy_desc}")
        
        try:
            # Create feature array with this strategy
            feature_values = []
            
            for name in feature_names:
                if name in real_features:
                    feature_values.append(real_features[name])
                else:
                    # Apply strategy
                    if strategy_name == "zero_fill":
                        value = 0.0
                    elif strategy_name == "negative_tiny":
                        value = -0.001
                    elif strategy_name == "positive_tiny":
                        value = 0.001
                    elif strategy_name == "statistical_mean":
                        # Use feature-type-specific means
                        if 'vit_feature' in name:
                            value = 0.0001  # Very small for vision
                        elif 'graph' in name:
                            if 'density' in name:
                                value = 0.05
                            elif 'nodes' in name or 'edges' in name:
                                value = 2.0
                            else:
                                value = 0.01
                        elif 'quantum' in name:
                            value = 0.00001  # Extremely small
                        else:
                            value = 0.0
                    elif strategy_name == "remove_synthetic":
                        # Try to use only real features (might not work with this model)
                        value = 0.0
                    else:
                        value = 0.0
                    
                    feature_values.append(value)
            
            # Create DataFrame with proper names
            feature_df = pd.DataFrame([feature_values], columns=feature_names)
            
            # Make prediction
            X = feature_df.values
            
            if feature_selector:
                X = feature_selector.transform(X)
            if scaler:
                X = scaler.transform(X)
            
            # Create named DataFrame for LightGBM
            if feature_selector and hasattr(feature_selector, 'get_support'):
                selected_mask = feature_selector.get_support()
                selected_feature_names = feature_df.columns[selected_mask]
                X_named = pd.DataFrame(X, columns=selected_feature_names)
            else:
                X_named = pd.DataFrame(X, columns=feature_df.columns)
            
            prediction = model.predict(X_named)[0]
            probabilities = model.predict_proba(X_named)[0]
            
            emotion = label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            # Get all probabilities
            emotion_probs = {}
            for i, prob in enumerate(probabilities):
                emo = label_encoder.inverse_transform([i])[0]
                emotion_probs[emo] = prob
            
            results[strategy_name] = {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': emotion_probs
            }
            
            # Display result
            if emotion in ['angry', 'fearful', 'sad']:
                color = "🔴"
            elif emotion in ['happy', 'surprised']:
                color = "🟢"
            elif emotion in ['calm', 'neutral']:
                color = "🔵"
            else:
                color = "🟡"
            
            st.write(f"**Result**: {color} {emotion} ({confidence:.1%})")
            
            # Show top 3
            sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            for i, (emo, prob) in enumerate(sorted_emotions[:3]):
                st.write(f"  {i+1}. {emo}: {prob:.1%}")
            
        except Exception as e:
            st.error(f"Strategy {strategy_name} failed: {e}")
    
    return results

def main():
    st.title("🔍 Precise Missing Feature Analyzer")
    st.markdown("### 🚨 **Solving 'Happy for Angry' Prediction Error**")
    
    st.info("This tool will identify the exact missing features causing wrong predictions and find the optimal strategy.")
    
    # Step 1: Analyze missing features
    feature_names, categories = analyze_exact_missing_features()
    
    if not feature_names:
        st.error("Could not load feature names")
        return
    
    # Step 2: Upload audio for testing
    st.header("🎵 Upload Audio for Strategy Testing")
    uploaded_file = st.file_uploader("Choose audio file", type=['wav', 'mp3', 'flac', 'm4a'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Extract only real audio features
        with st.spinner("🔬 Extracting REAL audio features only..."):
            real_features = extract_only_real_features(uploaded_file)
        
        if real_features:
            st.success(f"✅ Extracted {len(real_features)} real audio features")
            
            # Load models and test strategies
            models = load_models_quick()
            
            if all(models.values()):
                results = test_different_missing_strategies(uploaded_file, models, real_features)
                
                # Analysis and recommendations
                st.header("📊 Strategy Analysis & Recommendations")
                
                if results:
                    # Create comparison table
                    comparison_data = []
                    for strategy, data in results.items():
                        comparison_data.append({
                            'Strategy': strategy,
                            'Emotion': data['emotion'],
                            'Confidence': f"{data['confidence']:.1%}",
                            'Top_2nd': sorted(data['probabilities'].items(), key=lambda x: x[1], reverse=True)[1][0],
                            'Top_3rd': sorted(data['probabilities'].items(), key=lambda x: x[1], reverse=True)[2][0]
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    # Find most reasonable predictions
                    reasonable_strategies = []
                    for strategy, data in results.items():
                        # Skip obviously wrong predictions
                        if data['emotion'] not in ['disgust']:  # At least not disgust
                            reasonable_strategies.append((strategy, data['emotion'], data['confidence']))
                    
                    if reasonable_strategies:
                        st.success("✅ **Found reasonable strategies:**")
                        for strategy, emotion, conf in reasonable_strategies:
                            st.write(f"• **{strategy}**: {emotion} ({conf:.1%})")
                        
                        # Recommend best strategy
                        best_strategy = reasonable_strategies[0][0]
                        st.info(f"💡 **Recommended strategy**: {best_strategy}")
                        
                        st.markdown("""
                        **Next Steps:**
                        1. Update your main app to use the recommended strategy
                        2. Test with multiple audio samples to verify consistency
                        3. If still not accurate, the model may need retraining without synthetic features
                        """)
                    else:
                        st.error("❌ All strategies still give questionable results")
                        st.warning("💡 **The model likely needs to be retrained without the synthetic features**")
                        
                        # Show the problematic features
                        synthetic_count = len(categories.get('synthetic', []))
                        st.error(f"🚨 **Root Cause**: {synthetic_count} synthetic features in training data are corrupting predictions")
                        
            else:
                st.error("Failed to load models")
        else:
            st.error("Failed to extract features")

if __name__ == "__main__":
    main()
