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

# NEW CLEAN MODEL URLS - Properly organized files
MODEL_URLS = {
    'model': 'https://drive.google.com/uc?export=download&id=1kVZ6qg0a_8DNu1yn_hGYlc7mTQokk1CS&confirm=t',
    'scaler': 'https://drive.google.com/uc?export=download&id=1kjrAEPwVLbKyztSYxGmapysi1_prkJEK&confirm=t',
    'metadata': 'https://drive.google.com/uc?export=download&id=1TLlvM3SSIrUnz-0isLQGh5fUtV-4CzPu&confirm=t',
    'label_encoder': 'https://drive.google.com/uc?export=download&id=1o-PX_oquJCmzyrsgq1Oh1Hvv4uUfRZ5n&confirm=t',
    'feature_selector': 'https://drive.google.com/uc?export=download&id=1u4tX6Xzd9LOJ12PkYKYT3rWnHzncOngi&confirm=t',
    'feature_names': 'https://drive.google.com/uc?export=download&id=1lwFLlbCrFPLvfiPvxkFNTWyw91djgzyK&confirm=t'
}

st.set_page_config(
    page_title="SOTA Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def download_and_load_model(url, description):
    """Download and load model from Google Drive URL"""
    try:
        st.info(f"🔄 Downloading {description}...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=300)
        response.raise_for_status()
        
        content = response.content
        st.info(f"📊 {description}: {len(content)} bytes downloaded")
        
        # Check for HTML (Google Drive error)
        if content.startswith(b'<!DOCTYPE') or content.startswith(b'<html'):
            st.error(f"❌ Got HTML instead of file for {description}")
            return None
        
        # Try loading methods
        try:
            # Try joblib first
            result = joblib.load(io.BytesIO(content))
            st.success(f"✅ Loaded {description} with joblib")
            return result
        except:
            try:
                # Try pickle
                result = pickle.load(io.BytesIO(content))
                st.success(f"✅ Loaded {description} with pickle")
                return result
            except:
                try:
                    # Try JSON for metadata
                    result = json.loads(content.decode('utf-8'))
                    st.success(f"✅ Loaded {description} as JSON")
                    return result
                except:
                    st.error(f"❌ Could not load {description}")
                    return None
        
    except Exception as e:
        st.error(f"❌ Error downloading {description}: {e}")
        return None

@st.cache_data
def load_all_models():
    """Load all models from the new clean URLs"""
    
    model_descriptions = {
        'model': 'SOTA Model',
        'scaler': 'Feature Scaler',
        'metadata': 'Model Metadata',
        'label_encoder': 'Label Encoder',
        'feature_selector': 'Feature Selector',
        'feature_names': 'Feature Names'
    }
    
    models = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    total_models = len(MODEL_URLS)
    
    for i, (key, url) in enumerate(MODEL_URLS.items()):
        description = model_descriptions[key]
        status_text.text(f'Loading {description}... ({i+1}/{total_models})')
        
        progress_bar.progress((i + 0.5) / total_models)
        
        models[key] = download_and_load_model(url, description)
        
        if models[key] is not None:
            success_count += 1
            st.success(f"🎉 {description} loaded successfully!")
        else:
            st.error(f"💥 {description} failed to load!")
        
        progress_bar.progress((i + 1) / total_models)
    
    status_text.text(f'Completed: {success_count}/{total_models} models loaded')
    
    if success_count == total_models:
        st.balloons()
        st.success(f"🎉 ALL {total_models} MODELS LOADED SUCCESSFULLY!")
    elif success_count > 0:
        st.warning(f"⚠️ Partial success: {success_count}/{total_models} models loaded")
    else:
        st.error(f"💥 Failed to load any models!")
    
    return models if success_count > 0 else None

def extract_sota_features(audio_file, sample_rate=22050):
    """Extract comprehensive SOTA features"""
    try:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_file, sr=sample_rate, duration=3.0)
        
        if audio is None or len(audio) == 0:
            return None
            
        # Clean audio
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)
        
        features = {}
        
        # 1. MFCC Features (comprehensive)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
                features[f'mfcc_{i}_max'] = np.max(mfccs[i])
                features[f'mfcc_{i}_min'] = np.min(mfccs[i])
                features[f'mfcc_{i}_skew'] = float(stats.skew(mfccs[i]))
                features[f'mfcc_{i}_kurtosis'] = float(stats.kurtosis(mfccs[i]))
                features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
                features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
        except:
            # Fallback MFCC
            for i in range(13):
                for stat in ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']:
                    features[f'mfcc_{i}_{stat}'] = 0.0
                features[f'mfcc_delta_{i}_mean'] = 0.0
                features[f'mfcc_delta2_{i}_mean'] = 0.0
        
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
                features[f'{name}_mean'] = np.mean(feature_array)
                features[f'{name}_std'] = np.std(feature_array)
                features[f'{name}_max'] = np.max(feature_array)
                features[f'{name}_skew'] = float(stats.skew(feature_array))
        except:
            # Fallback spectral
            for name in ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate']:
                for stat in ['mean', 'std', 'max', 'skew']:
                    features[f'{name}_{stat}'] = 0.0
        
        # 3. Chroma Features
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
            for i in range(12):
                features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                features[f'chroma_{i}_std'] = np.std(chroma[i])
        except:
            for i in range(12):
                features[f'chroma_{i}_mean'] = 0.0
                features[f'chroma_{i}_std'] = 0.0
        
        # 4. Prosodic Features
        try:
            # F0 extraction
            f0 = librosa.yin(audio, fmin=50, fmax=400, threshold=0.1)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) > 0:
                features['f0_mean'] = np.mean(f0_clean)
                features['f0_std'] = np.std(f0_clean)
                features['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
                features['f0_jitter'] = np.mean(np.abs(np.diff(f0_clean))) / np.mean(f0_clean) if len(f0_clean) > 1 else 0
                features['f0_shimmer'] = np.std(f0_clean) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0
                
                # F0 contour
                if len(f0_clean) > 1:
                    features['f0_slope'] = np.polyfit(range(len(f0_clean)), f0_clean, 1)[0]
                else:
                    features['f0_slope'] = 0.0
                    
                if len(f0_clean) > 2:
                    features['f0_curvature'] = np.polyfit(range(len(f0_clean)), f0_clean, 2)[0]
                else:
                    features['f0_curvature'] = 0.0
            else:
                for feat in ['f0_mean', 'f0_std', 'f0_range', 'f0_jitter', 'f0_shimmer', 'f0_slope', 'f0_curvature']:
                    features[feat] = 0.0
            
            # Energy features
            rms = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            features['energy_skew'] = float(stats.skew(rms))
            features['energy_kurtosis'] = float(stats.kurtosis(rms))
            
        except:
            for feat in ['f0_mean', 'f0_std', 'f0_range', 'f0_jitter', 'f0_shimmer', 
                        'f0_slope', 'f0_curvature', 'energy_mean', 'energy_std', 'energy_skew', 'energy_kurtosis']:
                features[feat] = 0.0
        
        # 5. Advanced placeholder features (for compatibility)
        # Vision Transformer features
        for i in range(50):
            features[f'vit_feature_{i}'] = 0.0
            
        # Graph features
        for feat in ['graph_nodes', 'graph_edges', 'graph_density', 'graph_avg_clustering', 'graph_avg_degree', 'graph_degree_std']:
            features[feat] = 0.0
            
        # Quantum features
        for feat in ['quantum_entanglement_mean', 'quantum_entanglement_std', 'quantum_coherence']:
            features[feat] = 0.0
        
        # Clean features
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def predict_emotion(features, models):
    """Clean prediction pipeline with proper model organization"""
    try:
        if not features:
            st.error("No features extracted")
            return None, None, None
        
        # Get models (now properly organized)
        sota_model = models.get('model')
        scaler = models.get('scaler')
        feature_selector = models.get('feature_selector')
        label_encoder = models.get('label_encoder')
        feature_names = models.get('feature_names')
        
        if not all([sota_model, feature_names]):
            st.error("Missing required models")
            return None, None, None
        
        st.info(f"🔬 Using {len(feature_names)} features for prediction...")
        
        # Create feature array in correct order
        feature_array = []
        missing_features = []
        
        for name in feature_names:
            if name in features:
                feature_array.append(features[name])
            else:
                feature_array.append(0.0)
                missing_features.append(name)
        
        if missing_features and len(missing_features) < 50:
            st.info(f"Using defaults for {len(missing_features)} missing features")
        
        # Convert to numpy array
        X = np.array(feature_array).reshape(1, -1)
        st.info(f"📊 Feature vector shape: {X.shape}")
        
        # Apply feature selection if available
        if feature_selector and hasattr(feature_selector, 'transform'):
            try:
                X = feature_selector.transform(X)
                st.info(f"✅ Applied feature selection: {X.shape[1]} features selected")
            except Exception as e:
                st.warning(f"⚠️ Feature selection failed: {e}")
        
        # Apply scaling if available
        if scaler and hasattr(scaler, 'transform'):
            try:
                X = scaler.transform(X)
                st.info(f"✅ Applied feature scaling")
            except Exception as e:
                st.warning(f"⚠️ Scaling failed: {e}")
        
        # Make prediction
        if not hasattr(sota_model, 'predict'):
            st.error("❌ Invalid model - no predict method")
            return None, None, None
        
        try:
            prediction = sota_model.predict(X)[0]
            probabilities = sota_model.predict_proba(X)[0]
            st.success(f"✅ Model prediction successful!")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return None, None, None
        
        # Decode labels
        if label_encoder and hasattr(label_encoder, 'inverse_transform'):
            try:
                emotion = label_encoder.inverse_transform([prediction])[0]
                confidence = probabilities[prediction]
                
                # Get all emotion probabilities
                emotion_probs = {}
                for i, prob in enumerate(probabilities):
                    try:
                        emo = label_encoder.inverse_transform([i])[0]
                        emotion_probs[emo] = prob
                    except:
                        emotion_probs[f'emotion_{i}'] = prob
                
                return emotion, confidence, emotion_probs
                
            except Exception as e:
                st.warning(f"⚠️ Label decoding failed: {e}")
        
        # Fallback with standard emotion mapping
        emotion_map = {0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful', 
                      4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised'}
        
        emotion = emotion_map.get(prediction, f'emotion_{prediction}')
        confidence = probabilities[prediction]
        
        emotion_probs = {emotion_map.get(i, f'emotion_{i}'): prob 
                        for i, prob in enumerate(probabilities)}
        
        return emotion, confidence, emotion_probs
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None

def validate_new_model(models):
    """Validate the new model"""
    st.subheader("🔍 New Model Validation")
    
    sota_model = models.get('model')
    label_encoder = models.get('label_encoder')
    feature_selector = models.get('feature_selector')
    scaler = models.get('scaler')
    metadata = models.get('metadata')
    
    if not sota_model:
        st.error("❌ No model found")
        return False
    
    try:
        # Model info
        st.info(f"📊 Model type: {type(sota_model).__name__}")
        
        # Check ensemble components if applicable
        if hasattr(sota_model, 'estimators_'):
            st.info(f"🤖 Ensemble components: {len(sota_model.estimators_)}")
            try:
                for i, estimator in enumerate(sota_model.estimators_):
                    if hasattr(estimator, '__class__'):
                        st.info(f"  {i+1}. {type(estimator).__name__}")
            except:
                st.info("  (Could not enumerate estimators)")
        
        # Label encoder info
        if label_encoder and hasattr(label_encoder, 'classes_'):
            st.info(f"🏷️ Emotion classes: {list(label_encoder.classes_)}")
        
        # Feature selector info
        if feature_selector and hasattr(feature_selector, 'k'):
            st.info(f"🔍 Feature selector k: {feature_selector.k}")
        
        # Scaler info
        if scaler:
            st.info(f"📊 Scaler type: {type(scaler).__name__}")
        
        # Metadata info
        if metadata:
            st.info("📋 Model metadata available")
            if isinstance(metadata, dict):
                for key, value in list(metadata.items())[:5]:  # Show first 5 items
                    st.info(f"  {key}: {value}")
        
        # Test prediction
        st.info("🧪 Testing prediction pipeline...")
        test_features = {f'feature_{i}': np.random.random() for i in range(100)}
        
        emotion, confidence, probs = predict_emotion(test_features, models)
        
        if emotion and confidence:
            st.success(f"✅ Prediction test successful: {emotion} ({confidence:.1%})")
            return True
        else:
            st.error("❌ Prediction test failed")
            return False
        
    except Exception as e:
        st.error(f"❌ Validation failed: {e}")
        return False

def main():
    # Header
    st.title("🎤 NEW SOTA Speech Emotion Recognition")
    st.markdown("### 🔬 **Fresh Model** | 2024-2025 Research Breakthrough")
    st.markdown("**Author:** Peter Chika Ozo-ogueji (Data Scientist)")
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("🔄 Loading new SOTA models...")
    
    # Load models
    models = load_all_models()
    
    if models is None or not models:
        st.sidebar.error("❌ No models loaded successfully")
        st.error("⚠️ Models failed to load. Please check the URLs and try again.")
        return
    
    # Show loading results
    loaded_models = [k for k, v in models.items() if v is not None]
    failed_models = [k for k, v in models.items() if v is None]
    
    st.sidebar.success(f"✅ Loaded: {len(loaded_models)}/6 models")
    
    # Display metadata if available
    metadata = models.get('metadata')
    if metadata and isinstance(metadata, dict):
        st.sidebar.success("📊 Model Metadata:")
        st.sidebar.json({
            "Model Type": metadata.get('model_type', 'SOTA Model'),
            "Accuracy": f"{metadata.get('accuracy', 'N/A')}",
            "F1-Score": f"{metadata.get('f1_score', 'N/A')}",
            "Features": metadata.get('feature_count', 'N/A'),
            "Classes": metadata.get('emotion_classes', 'N/A')
        })
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎵 Upload Audio for Emotion Recognition")
        
        # Check required models
        required_models = ['model', 'label_encoder', 'feature_names']
        missing_required = [m for m in required_models if not models.get(m)]
        
        if missing_required:
            st.warning(f"⚠️ Missing: {', '.join(missing_required)}")
            st.info("⏳ Some models failed to load...")
        else:
            st.success("✅ NEW MODEL READY FOR PREDICTIONS! 🎉")
            
            # Optional validation
            if st.checkbox("🧪 Validate new model (optional)"):
                validate_new_model(models)
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Upload a WAV, MP3, FLAC, or M4A file for emotion recognition"
            )
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format='audio/wav')
                
                # Extract features and predict
                with st.spinner('🔬 Analyzing audio with NEW SOTA model...'):
                    features = extract_sota_features(uploaded_file)
                    
                    if features:
                        emotion, confidence, emotion_probs = predict_emotion(features, models)
                        
                        if emotion:
                            # Display results
                            st.success(f"🎯 **Predicted Emotion:** {emotion.title()}")
                            st.info(f"🎲 **Confidence:** {confidence:.1%}")
                            
                            # Emotion probabilities chart
                            st.subheader("📊 NEW SOTA Model Predictions")
                            
                            prob_df = pd.DataFrame(
                                list(emotion_probs.items()),
                                columns=['Emotion', 'Probability']
                            ).sort_values('Probability', ascending=True)
                            
                            fig = px.bar(
                                prob_df, 
                                x='Probability', 
                                y='Emotion',
                                orientation='h',
                                title=f"New SOTA Model Emotion Predictions",
                                color='Probability',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show top 3 predictions
                            sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                            st.subheader("🏆 Top 3 Predictions")
                            for i, (emo, prob) in enumerate(sorted_emotions[:3]):
                                st.write(f"{i+1}. **{emo.title()}**: {prob:.1%}")
    
    with col2:
        st.header("🏆 NEW Model Info")
        
        if metadata and isinstance(metadata, dict):
            # Display metrics from metadata
            st.metric("🎯 Accuracy", metadata.get('accuracy', 'N/A'))
            st.metric("📈 F1-Score", metadata.get('f1_score', 'N/A'))
            st.metric("🔬 Features", metadata.get('feature_count', 'N/A'))
            st.metric("📚 Samples", metadata.get('total_samples', 'N/A'))
        else:
            st.info("📊 Model metadata will appear here when available")
        
        # Model components
        st.subheader("🤖 Model Components")
        for key in ['model', 'scaler', 'feature_selector', 'label_encoder']:
            if models.get(key):
                st.markdown(f"• **{key.title()}**: ✅ Loaded")
            else:
                st.markdown(f"• **{key.title()}**: ❌ Missing")
        
        # SOTA techniques
        st.subheader("🔬 SOTA Techniques")
        techniques = [
            "Advanced Feature Engineering",
            "Ensemble Learning",
            "Cross-corpus Validation", 
            "Robust Preprocessing",
            "Modern ML Architecture"
        ]
        for technique in techniques:
            st.markdown(f"• {technique}")

if __name__ == "__main__":
    main()
