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
import sys

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
    page_title="🔧 Version-Compatible Model Loader",
    page_icon="🔧",
    layout="wide"
)

def try_all_loading_methods(content, description):
    """Try every possible method to load the pickle file"""
    
    loading_methods = [
        # Method 1: Standard joblib
        lambda: joblib.load(io.BytesIO(content)),
        
        # Method 2: Standard pickle
        lambda: pickle.load(io.BytesIO(content)),
        
        # Method 3: Pickle with different protocols
        lambda: pickle.load(io.BytesIO(content), encoding='latin1'),
        lambda: pickle.load(io.BytesIO(content), encoding='bytes'),
        
        # Method 4: Joblib with different parameters
        lambda: joblib.load(io.BytesIO(content), mmap_mode=None),
        
        # Method 5: Try older pickle protocols
        lambda: pickle.loads(content, encoding='latin1'),
        lambda: pickle.loads(content, encoding='bytes'),
        lambda: pickle.loads(content),
    ]
    
    errors = []
    
    for i, method in enumerate(loading_methods):
        try:
            result = method()
            st.success(f"✅ {description}: Loaded with method {i+1}")
            return result
        except Exception as e:
            errors.append(f"Method {i+1}: {str(e)[:100]}")
            continue
    
    # If all methods failed, show detailed error info
    st.error(f"❌ {description}: All loading methods failed")
    with st.expander(f"Show {description} loading errors"):
        for error in errors:
            st.text(error)
    
    return None

@st.cache_data
def load_models_with_compatibility():
    """Load models with maximum compatibility"""
    
    st.info("🔧 Attempting to load models with version compatibility fixes...")
    
    # Show environment info
    st.info(f"Python version: {sys.version}")
    st.info(f"Scikit-learn version: {__import__('sklearn').__version__}")
    st.info(f"Joblib version: {joblib.__version__}")
    
    models = {}
    
    for key, url in MODEL_URLS.items():
        st.write(f"🔄 Loading {key}...")
        
        try:
            # Download file
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=300)
            response.raise_for_status()
            content = response.content
            
            st.info(f"📥 Downloaded {len(content)} bytes for {key}")
            
            # Check for HTML errors
            if content.startswith(b'<!DOCTYPE') or b'<title>Google Drive' in content[:500]:
                st.error(f"❌ {key}: Received HTML error page instead of file")
                models[key] = None
                continue
            
            # Special handling for JSON metadata
            if key == 'metadata':
                try:
                    models[key] = json.loads(content.decode('utf-8'))
                    st.success(f"✅ {key}: Loaded as JSON")
                    continue
                except Exception as e:
                    st.error(f"❌ {key}: JSON parsing failed - {e}")
                    models[key] = None
                    continue
            
            # For pickle files, try all compatibility methods
            models[key] = try_all_loading_methods(content, key)
            
            # Show additional info about loaded objects
            if models[key] is not None:
                obj = models[key]
                st.info(f"🔍 {key} type: {type(obj).__name__}")
                
                if hasattr(obj, 'shape'):
                    st.info(f"🔍 {key} shape: {obj.shape}")
                elif hasattr(obj, '__len__') and key == 'feature_names':
                    st.info(f"🔍 {key} length: {len(obj)}")
                elif hasattr(obj, 'classes_') and key == 'label_encoder':
                    st.info(f"🔍 {key} classes: {list(obj.classes_)}")
                
        except Exception as e:
            st.error(f"❌ {key}: Download or processing failed - {e}")
            models[key] = None
    
    # Summary
    loaded_count = sum(1 for v in models.values() if v is not None)
    st.info(f"📊 Successfully loaded {loaded_count}/6 components")
    
    return models

def create_fallback_components():
    """Create fallback components if models can't be loaded"""
    
    st.warning("🔧 Creating fallback components for testing...")
    
    # Create a simple label encoder mapping
    emotion_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    class FallbackLabelEncoder:
        def __init__(self):
            self.classes_ = np.array(emotion_classes)
            self.class_to_index = {cls: i for i, cls in enumerate(emotion_classes)}
        
        def transform(self, y):
            return np.array([self.class_to_index[cls] for cls in y])
        
        def inverse_transform(self, y):
            return np.array([self.classes_[idx] for idx in y])
    
    # Create basic feature names (first 100 features from what we know work)
    basic_features = []
    
    # MFCC features
    for i in range(13):
        for stat in ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']:
            basic_features.append(f'mfcc_{i}_{stat}')
        basic_features.append(f'mfcc_delta_{i}_mean')
        basic_features.append(f'mfcc_delta2_{i}_mean')
    
    # Spectral features
    for name in ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate']:
        for stat in ['mean', 'std', 'max', 'skew']:
            basic_features.append(f'{name}_{stat}')
    
    # Chroma features
    for i in range(12):
        basic_features.append(f'chroma_{i}_mean')
        basic_features.append(f'chroma_{i}_std')
    
    fallback_models = {
        'label_encoder': FallbackLabelEncoder(),
        'feature_names': basic_features,
        'metadata': {
            'model_type': 'Fallback Test Model',
            'accuracy': 'N/A',
            'f1_score': 'N/A',
            'emotion_classes': emotion_classes
        }
    }
    
    st.success("✅ Created fallback components for basic testing")
    return fallback_models

def extract_basic_features(audio_file, feature_names):
    """Extract basic audio features that work reliably"""
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=22050, duration=3.0)
        
        if audio is None or len(audio) == 0:
            return None
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)
        
        features = {}
        
        # Extract MFCC features
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
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
            st.warning("⚠️ MFCC extraction failed")
        
        # Extract spectral features
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
            st.warning("⚠️ Spectral feature extraction failed")
        
        # Extract chroma features
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            for i in range(12):
                features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                features[f'chroma_{i}_std'] = np.std(chroma[i])
        except:
            st.warning("⚠️ Chroma feature extraction failed")
        
        # Fill in missing features with zeros
        for name in feature_names:
            if name not in features:
                features[name] = 0.0
        
        # Clean features
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        return features
        
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

def main():
    st.title("🔧 Version-Compatible Model Loader")
    st.markdown("### 🚨 **Fixing 'Invalid Load Key' Errors**")
    
    st.error("**DETECTED ISSUE**: Your model files have version compatibility problems!")
    st.info("This tool will try multiple loading methods to fix the 'invalid load key' errors.")
    
    # Try to load models with compatibility fixes
    st.header("📥 Loading Models with Compatibility Fixes")
    models = load_models_with_compatibility()
    
    loaded_count = sum(1 for v in models.values() if v is not None)
    
    if loaded_count < 2:
        st.error("❌ Critical model files couldn't be loaded - using fallback components")
        st.info("💡 **Recommendation**: Your model files were likely saved with a different version of scikit-learn or Python.")
        st.info("🔧 **Solutions**: 1) Re-save models with current environment, 2) Use environment matching the original training")
        
        # Create fallback for basic testing
        models = create_fallback_components()
        st.warning("⚠️ Using fallback components - predictions won't be from your actual model")
    
    # Show what was loaded
    st.subheader("📊 Loading Results")
    for key, obj in models.items():
        if obj is not None:
            st.success(f"✅ {key}: Loaded successfully")
        else:
            st.error(f"❌ {key}: Failed to load")
    
    # Test with audio if we have basic components
    if models.get('feature_names') and models.get('label_encoder'):
        st.header("🎵 Test Audio Processing")
        
        uploaded_file = st.file_uploader(
            "Upload audio to test feature extraction",
            type=['wav', 'mp3', 'flac', 'm4a']
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            with st.spinner("🔬 Extracting features..."):
                features = extract_basic_features(uploaded_file, models['feature_names'])
                
                if features:
                    st.success(f"✅ Extracted {len(features)} features")
                    
                    # Show feature sample
                    feature_sample = {k: v for k, v in list(features.items())[:10]}
                    st.json(feature_sample)
                    
                    st.info("✅ **Good News**: Feature extraction works!")
                    st.info("❌ **Bad News**: Your actual model files are corrupted/incompatible")
                    
                else:
                    st.error("❌ Feature extraction failed")
    
    # Recommendations
    st.header("💡 Recommendations")
    
    if loaded_count == 0:
        st.error("🚨 **CRITICAL**: No model files could be loaded")
        st.markdown("""
        **Immediate Actions:**
        1. **Check Python/scikit-learn versions** - Your model was likely saved with different versions
        2. **Re-download model files** - They might be corrupted
        3. **Contact model creator** - Ask for files saved with your environment versions
        4. **Try different environment** - Use the exact Python/scikit-learn versions used for training
        """)
        
    elif loaded_count < 4:
        st.warning("⚠️ **PARTIAL FAILURE**: Some model files loaded, others didn't")
        st.markdown("""
        **Likely Causes:**
        - Version mismatch between training and inference environments
        - Partial file corruption
        - Mixed serialization methods (some with joblib, some with pickle)
        
        **Solutions:**
        1. **Environment matching**: Use same Python + scikit-learn versions as training
        2. **Re-save models**: Have original creator re-save with current versions
        3. **Check file integrity**: Re-download files to ensure no corruption
        """)
        
    else:
        st.success("✅ **SUCCESS**: Most files loaded - ready for testing!")
        st.info("Your models should now work correctly for predictions")
    
    # Version compatibility info
    with st.expander("🔍 Version Compatibility Details"):
        st.markdown("""
        **Common Causes of 'Invalid Load Key' Errors:**
        
        1. **Python Version Mismatch**: Models saved with Python 3.8 but loading with Python 3.9+
        2. **Scikit-learn Version**: Models saved with scikit-learn 0.24 but loading with 1.0+
        3. **Joblib Version**: Different joblib versions use different protocols
        4. **Platform Differences**: Models saved on Windows but loading on Linux/Mac
        5. **Pickle Protocol**: Models saved with newer pickle protocol than current Python supports
        
        **Solutions:**
        - Match exact versions used during training
        - Re-save models with compatible versions
        - Use `pickle.HIGHEST_PROTOCOL` when saving
        - Save with `joblib.dump(model, 'file.pkl', protocol=2)` for compatibility
        """)

if __name__ == "__main__":
    main()
