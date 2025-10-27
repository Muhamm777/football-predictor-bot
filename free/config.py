import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API ключи
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY', 'your-api-key-here')
    
    # Настройки базы данных
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///football_predictions.db')
    
    # Настройки прогнозирования
    PREDICTION_CONFIDENCE_THRESHOLD = 0.6
    MIN_MATCHES_FOR_PREDICTION = 5
    
    # Настройки веб-интерфейса
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    DEBUG = True
    
    # Настройки уведомлений
    ENABLE_NOTIFICATIONS = True
    NOTIFICATION_TIME = "09:00"  # Время отправки прогнозов
