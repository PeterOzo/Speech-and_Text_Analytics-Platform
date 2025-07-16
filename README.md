# 🎙️ AnalyticsPro - Speech Emotion Recognition

Professional Speech Emotion Recognition Platform with 82%+ Accuracy

## 🌟 Features

- **Real-time Emotion Detection** from audio files
- **8 Emotion Classes**: Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised
- **191 Clean Audio Features** (No synthetic features)
- **82.0% Test Accuracy** / **83.1% F1-Score**
- **Batch Processing** for multiple files
- **Advanced Analytics Dashboard**
- **Professional Visualizations**

## 🚀 Live Demo

[Visit AnalyticsPro](https://your-app-url.streamlit.app)

## 🛠️ Technology Stack

- **Model**: Ensemble (XGBoost + LightGBM + Random Forest + SVM)
- **Frontend**: Streamlit
- **Audio Processing**: Librosa
- **Visualization**: Plotly
- **ML Framework**: Scikit-learn

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 82.0% |
| F1-Score | 83.1% |
| Features | 191 |
| Training Samples | 10,982 |

## 🔧 Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/analyticspro.git
cd analyticspro

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- M4A
- AAC

## 👨‍💻 Author

**Peter Chika Ozo-ogueji** (Data Scientist)

## 📄 License

© 2024 AnalyticsPro. All rights reserved.

## 🙏 Acknowledgments

Model trained on benchmark datasets:
- RAVDESS (85%+ accuracy)
- CREMA-D (80%+ validated)
- TESS (High performance)
- EMO-DB (90%+ on clean features)
- SAVEE (Proven benchmark)
