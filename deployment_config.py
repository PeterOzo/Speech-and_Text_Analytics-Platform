import os
from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    
    # App Settings
    APP_TITLE = "Speech Analytics Platform - American Credit Acceptance"
    APP_ICON = "🎯"
    
    # Performance Settings
    MAX_UPLOAD_SIZE = 50  # MB
    PROCESSING_TIMEOUT = 30  # seconds
    
    # Business Settings
    COMPANY_NAME = "American Credit Acceptance LLC"
    TARGET_ACCURACY = 82.4
    
    # Feature Flags
    ENABLE_REAL_TIME_PROCESSING = True
    ENABLE_AUDIO_UPLOAD = True
    ENABLE_BATCH_PROCESSING = False  # For demo
    
    # URLs
    GITHUB_REPO = "https://github.com/PeterOzo/speech-analytics-platform"
    LINKEDIN_PROFILE = "http://linkedin.com/in/peterchika/"
    
    # Environment-specific settings
    DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Business Metrics
    MONTHLY_COST_SAVINGS = 120000  # $120K monthly
    MONTHLY_REVENUE_IMPACT = 387000  # $387K monthly
    ANNUAL_ROI = 7584000  # $7.584M annual
    
    # Alert Thresholds
    ESCALATION_THRESHOLD = 0.7
    SATISFACTION_THRESHOLD = 3.5
    CRITICAL_ALERT_THRESHOLD = 0.9
    
    # Model Performance
    MODEL_ACCURACY = 82.4
    F1_SCORE = 83.5
    PROCESSING_SPEED = 1.2  # seconds per call
    
    @classmethod
    def get_environment(cls):
        """Get current deployment environment"""
        return os.getenv('STREAMLIT_ENV', 'production')
    
    @classmethod
    def is_production(cls):
        """Check if running in production"""
        return cls.get_environment() == 'production'
    
    @classmethod
    def get_app_url(cls):
        """Get application URL based on environment"""
        if cls.is_production():
            return "https://speech-analytics-aca.streamlit.app/"
        return "http://localhost:8501"