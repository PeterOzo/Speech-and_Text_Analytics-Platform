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
    page_title="🔍 COMPREHENSIVE MODEL DIAGNOSTIC",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_models():
    """Load models with detailed validation"""
    models = {}
    
    for key, url in MODEL_URLS.items():
        try:
            response = requests.get(url, timeout=300)
            content = response.content
            
            if content.startswith(b'<!DOCTYPE') or b'<title>Google Drive' in content[:500]:
                st.error(f"❌ {key}: HTML error page received")
                models[key] = None
                continue
            
            # Try loading
            try:
                if key == 'metadata':
                    models[key] = json.loads(content.decode('utf-8'))
                else:
                    models[key] = pickle.load(io.BytesIO(content))
                st.success(f"✅ {key}: Loaded successfully")
                
                # Detailed inspection
                if key == 'model':
                    model = models[key]
                    st.info(f"Model type: {type(model).__name__}")
                    if hasattr(model, 'n_classes_'):
                        st.info(f"Model expects {model.n_classes_} classes")
                    if hasattr(model, 'n_features_'):
                        st.info(f"Model expects {model.n_features_} features")
                        
                elif key == 'label_encoder':
                    le = models[key]
                    if hasattr(le, 'classes_'):
                        st.info(f"Label classes: {list(le.classes_)}")
                        
                elif key == 'feature_names':
                    fn = models[key]
                    st.info(f"Feature count: {len(fn)}")
                    st.info(f"Sample features: {fn[:3]}...{fn[-3:]}")
                    
            except Exception as e:
                st.error(f"❌ {key}: Load failed - {e}")
                models[key] = None
                
        except Exception as e:
            st.error(f"❌ {key}: Download failed - {e}")
            models[key] = None
    
    return models

def test_model_sanity(models):
    """Test if the model makes reasonable predictions with controlled inputs"""
    
    st.subheader("🧪 Model Sanity Tests")
    
    model = models.get('model')
    scaler = models.get('scaler') 
    feature_selector = models.get('feature_selector')
    label_encoder = models.get('label_encoder')
    feature_names = models.get('feature_names')
    
    if not all([model, feature_names]):
        st.error("Missing required components for testing")
        return False
    
    # Test 1: All zeros
    st.write("**Test 1: All Zero Features**")
    test_features = np.zeros((1, len(feature_names)))
    
    try:
        X = test_features.copy()
        if feature_selector:
            X = feature_selector.transform(X)
        if scaler:
            X = scaler.transform(X)
        
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        if label_encoder:
            emotion = label_encoder.inverse_transform([pred])[0]
        else:
            emotion = f"class_{pred}"
            
        st.write(f"All zeros → {emotion} ({probs[pred]:.1%})")
        st.write(f"All probabilities: {[f'{p:.3f}' for p in probs]}")
        
        # If all zeros gives very high confidence, that's suspicious
        if probs[pred] > 0.8:
            st.warning("⚠️ SUSPICIOUS: High confidence on all-zero features!")
            
    except Exception as e:
        st.error(f"Test 1 failed: {e}")
        return False
    
    # Test 2: Random features in reasonable ranges
    st.write("**Test 2: Random Features (Audio-like ranges)**")
    np.random.seed(42)
    test_features = np.random.normal(0, 1, (1, len(feature_names)))
    
    # Make MFCC features more realistic
    for i, name in enumerate(feature_names):
        if 'mfcc' in name and 'mean' in name:
            test_features[0, i] = np.random.normal(-20, 10)  # Typical MFCC range
        elif 'spectral_centroid' in name:
            test_features[0, i] = np.random.normal(2000, 500)  # Typical spectral centroid
        elif 'f0' in name and 'mean' in name:
            test_features[0, i] = np.random.normal(150, 50)  # Typical F0
    
    try:
        X = test_features.copy()
        if feature_selector:
            X = feature_selector.transform(X)
        if scaler:
            X = scaler.transform(X)
        
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        if label_encoder:
            emotion = label_encoder.inverse_transform([pred])[0]
        else:
            emotion = f"class_{pred}"
            
        st.write(f"Realistic random → {emotion} ({probs[pred]:.1%})")
        st.write(f"All probabilities: {[f'{p:.3f}' for p in probs]}")
        
    except Exception as e:
        st.error(f"Test 2 failed: {e}")
        return False
    
    # Test 3: Extreme values
    st.write("**Test 3: Extreme Feature Values**")
    test_features = np.full((1, len(feature_names)), 100.0)  # Very high values
    
    try:
        X = test_features.copy()
        if feature_selector:
            X = feature_selector.transform(X)
        if scaler:
            X = scaler.transform(X)
        
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        if label_encoder:
            emotion = label_encoder.inverse_transform([pred])[0]
        else:
            emotion = f"class_{pred}"
            
        st.write(f"Extreme values → {emotion} ({probs[pred]:.1%})")
        st.write(f"All probabilities: {[f'{p:.3f}' for p in probs]}")
        
    except Exception as e:
        st.error(f"Test 3 failed: {e}")
        return False
    
    # Test 4: Check if model always predicts the same class
    st.write("**Test 4: Multiple Random Tests**")
    predictions = []
    for i in range(10):
        test_features = np.random.normal(0, 1, (1, len(feature_names)))
        
        try:
            X = test_features.copy()
            if feature_selector:
                X = feature_selector.transform(X)
            if scaler:
                X = scaler.transform(X)
            
            pred = model.predict(X)[0]
            predictions.append(pred)
        except:
            predictions.append(-1)
    
    unique_preds = set(predictions)
    st.write(f"10 random tests gave {len(unique_preds)} unique predictions: {unique_preds}")
    
    if len(unique_preds) == 1 and -1 not in unique_preds:
        st.error("🚨 MAJOR ISSUE: Model always predicts the same class!")
        return False
    elif len(unique_preds) < 3:
        st.warning("⚠️ Model shows limited diversity in predictions")
        
    return True

def test_label_encoder(models):
    """Test label encoder functionality"""
    
    st.subheader("🏷️ Label Encoder Test")
    
    label_encoder = models.get('label_encoder')
    if not label_encoder:
        st.error("No label encoder found")
        return False
    
    try:
        # Test encoding/decoding
        if hasattr(label_encoder, 'classes_'):
            classes = label_encoder.classes_
            st.write(f"Available classes: {list(classes)}")
            
            # Test each class
            st.write("**Encoding/Decoding Test:**")
            for i, class_name in enumerate(classes):
                # Test forward and backward
                encoded = label_encoder.transform([class_name])[0]
                decoded = label_encoder.inverse_transform([encoded])[0]
                
                st.write(f"{class_name} → {encoded} → {decoded} {'✅' if decoded == class_name else '❌'}")
                
                if decoded != class_name:
                    st.error(f"Label encoder broken for {class_name}")
                    return False
            
            # Test numeric indices
            st.write("**Index Mapping:**")
            for i in range(len(classes)):
                try:
                    decoded = label_encoder.inverse_transform([i])[0]
                    st.write(f"Index {i} → {decoded}")
                except Exception as e:
                    st.error(f"Index {i} failed: {e}")
                    return False
        
        return True
        
    except Exception as e:
        st.error(f"Label encoder test failed: {e}")
        return False

def analyze_feature_importance(models, features):
    """Analyze which features might be causing issues"""
    
    st.subheader("🔬 Feature Analysis")
    
    model = models.get('model')
    feature_names = models.get('feature_names')
    
    if not all([model, feature_names, features]):
        st.error("Missing components for feature analysis")
        return
    
    try:
        # Get feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            st.write("**Top 10 Most Important Features:**")
            for i, row in importance_df.head(10).iterrows():
                feature_name = row['feature']
                importance = row['importance']
                feature_value = features.get(feature_name, 'MISSING')
                st.write(f"{feature_name}: {importance:.4f} (value: {feature_value})")
            
            # Check if any critical features are missing or have unusual values
            critical_features = importance_df.head(20)
            issues = []
            
            for _, row in critical_features.iterrows():
                feature_name = row['feature']
                if feature_name not in features:
                    issues.append(f"Critical feature MISSING: {feature_name}")
                elif np.isnan(features[feature_name]) or np.isinf(features[feature_name]):
                    issues.append(f"Critical feature has invalid value: {feature_name} = {features[feature_name]}")
            
            if issues:
                st.error("🚨 Critical Feature Issues:")
                for issue in issues:
                    st.error(f"• {issue}")
        
    except Exception as e:
        st.error(f"Feature importance analysis failed: {e}")

def test_preprocessing_pipeline(models, features):
    """Test each step of the preprocessing pipeline"""
    
    st.subheader("⚙️ Preprocessing Pipeline Test")
    
    feature_names = models.get('feature_names')
    feature_selector = models.get('feature_selector')
    scaler = models.get('scaler')
    
    if not all([feature_names, features]):
        st.error("Missing components for pipeline test")
        return None
    
    # Step 1: Create feature array
    st.write("**Step 1: Feature Array Creation**")
    feature_array = []
    missing_count = 0
    
    for name in feature_names:
        if name in features:
            feature_array.append(features[name])
        else:
            feature_array.append(0.0)
            missing_count += 1
    
    X = np.array(feature_array).reshape(1, -1)
    st.write(f"Initial shape: {X.shape}")
    st.write(f"Missing features: {missing_count}")
    st.write(f"Value range: [{X.min():.3f}, {X.max():.3f}]")
    st.write(f"Mean: {X.mean():.3f}, Std: {X.std():.3f}")
    
    # Check for problematic values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        st.error("🚨 NaN or Inf values detected in features!")
        return None
    
    if X.std() == 0:
        st.warning("⚠️ All features have the same value - this is suspicious!")
    
    # Step 2: Feature selection
    if feature_selector:
        st.write("**Step 2: Feature Selection**")
        try:
            X_selected = feature_selector.transform(X)
            st.write(f"After selection: {X.shape} → {X_selected.shape}")
            st.write(f"Selected value range: [{X_selected.min():.3f}, {X_selected.max():.3f}]")
            
            # Check if feature selector is working properly
            if hasattr(feature_selector, 'get_support'):
                selected_mask = feature_selector.get_support()
                selected_indices = np.where(selected_mask)[0]
                st.write(f"Selected {len(selected_indices)} features out of {len(selected_mask)}")
                
                # Show some selected features
                if len(selected_indices) > 0:
                    selected_feature_names = [feature_names[i] for i in selected_indices[:10]]
                    st.write(f"Top selected features: {selected_feature_names}")
            
            X = X_selected
        except Exception as e:
            st.error(f"Feature selection failed: {e}")
            return None
    
    # Step 3: Scaling
    if scaler:
        st.write("**Step 3: Feature Scaling**")
        try:
            X_scaled = scaler.transform(X)
            st.write(f"After scaling: {X_scaled.shape}")
            st.write(f"Scaled value range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
            st.write(f"Scaled mean: {X_scaled.mean():.3f}, std: {X_scaled.std():.3f}")
            
            # Check if scaling seems reasonable
            if X_scaled.std() > 10:
                st.warning("⚠️ Very high standard deviation after scaling - might indicate issues")
            
            X = X_scaled
        except Exception as e:
            st.error(f"Scaling failed: {e}")
            return None
    
    return X

def main():
    st.title("🔍 COMPREHENSIVE MODEL DIAGNOSTIC")
    st.markdown("### 🚨 **Deep Debugging for Persistent 'Disgust' Predictions**")
    
    st.info("This tool will thoroughly test your model to find why it keeps predicting 'disgust'")
    
    # Load models
    st.header("📥 Model Loading")
    models = load_models()
    
    loaded_count = sum(1 for v in models.values() if v is not None)
    if loaded_count < 4:  # Need at least model, feature_names, label_encoder, and one other
        st.error(f"Only {loaded_count}/6 models loaded - cannot proceed")
        return
    
    st.success(f"✅ {loaded_count}/6 models loaded successfully")
    
    # Run sanity tests
    st.header("🧪 Model Sanity Tests")
    sanity_ok = test_model_sanity(models)
    
    # Test label encoder
    st.header("🏷️ Label Encoder Tests")
    label_ok = test_label_encoder(models)
    
    # Audio file upload for real testing
    st.header("🎵 Real Audio Testing")
    uploaded_file = st.file_uploader("Upload an audio file for testing", type=['wav', 'mp3', 'flac', 'm4a'])
    
    if uploaded_file is not None:
        # Extract features (simplified version)
        try:
            audio, sr = librosa.load(uploaded_file, sr=22050, duration=3.0)
            
            # Basic feature extraction
            features = {}
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
                features[f'mfcc_{i}_max'] = np.max(mfccs[i])
                features[f'mfcc_{i}_min'] = np.min(mfccs[i])
                features[f'mfcc_{i}_skew'] = float(stats.skew(mfccs[i]))
                features[f'mfcc_{i}_kurtosis'] = float(stats.kurtosis(mfccs[i]))
            
            # Fill remaining features with zeros or reasonable defaults
            feature_names = models.get('feature_names', [])
            for name in feature_names:
                if name not in features:
                    features[name] = 0.0
            
            st.success(f"Extracted {len(features)} features")
            
            # Analyze features
            analyze_feature_importance(models, features)
            
            # Test preprocessing pipeline
            final_X = test_preprocessing_pipeline(models, features)
            
            if final_X is not None:
                # Make prediction with detailed logging
                model = models.get('model')
                label_encoder = models.get('label_encoder')
                
                try:
                    pred = model.predict(final_X)[0]
                    probs = model.predict_proba(final_X)[0]
                    
                    st.write("**Final Prediction Results:**")
                    st.write(f"Raw prediction index: {pred}")
                    st.write(f"Raw probabilities: {[f'{p:.4f}' for p in probs]}")
                    
                    if label_encoder:
                        emotion = label_encoder.inverse_transform([pred])[0]
                        st.write(f"Decoded emotion: {emotion}")
                        
                        # Show all emotion probabilities
                        st.write("**All Emotion Probabilities:**")
                        for i, prob in enumerate(probs):
                            try:
                                emo = label_encoder.inverse_transform([i])[0]
                                st.write(f"  {emo}: {prob:.4f} ({prob*100:.1f}%)")
                            except:
                                st.write(f"  class_{i}: {prob:.4f} ({prob*100:.1f}%)")
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            
        except Exception as e:
            st.error(f"Audio processing failed: {e}")
    
    # Summary and recommendations
    st.header("📋 Diagnostic Summary")
    
    if not sanity_ok:
        st.error("🚨 **CRITICAL ISSUE**: Model fails basic sanity tests!")
        st.error("**Recommendation**: The model file may be corrupted or wrong. Try re-downloading or using a different model.")
    elif not label_ok:
        st.error("🚨 **CRITICAL ISSUE**: Label encoder is broken!")
        st.error("**Recommendation**: The label encoder file is corrupted. Re-download it.")
    else:
        st.success("✅ Basic components seem to work correctly")
        st.info("**Next Steps**: Upload audio files and check the detailed analysis above for specific issues.")

if __name__ == "__main__":
    main()
