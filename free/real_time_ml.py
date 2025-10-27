"""
Система машинного обучения в реальном времени для футбольных прогнозов
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import queue
import logging

class RealTimeMLSystem:
    """Система машинного обучения в реальном времени"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.data_queue = queue.Queue()
        self.is_running = False
        self.update_interval = 300  # 5 минут
        self.retrain_threshold = 0.05  # 5% снижение точности
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Инициализация моделей
        self._initialize_models()
    
    def _initialize_models(self):
        """Инициализация моделей"""
        self.models = {
            'primary': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=42
            ),
            'secondary': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=42
            ),
            'backup': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        # Инициализация производительности
        for model_name in self.models.keys():
            self.model_performance[model_name] = {
                'accuracy': 0.0,
                'last_update': datetime.now(),
                'predictions_count': 0,
                'correct_predictions': 0
            }
    
    def add_new_data(self, match_data: Dict):
        """Добавить новые данные в очередь"""
        match_data['timestamp'] = datetime.now()
        self.data_queue.put(match_data)
        self.logger.info(f"Добавлены новые данные: {match_data.get('match_id', 'unknown')}")
    
    def start_real_time_system(self):
        """Запустить систему реального времени"""
        self.is_running = True
        
        # Запускаем обработку данных в отдельном потоке
        data_thread = threading.Thread(target=self._process_data_queue)
        data_thread.daemon = True
        data_thread.start()
        
        # Запускаем мониторинг производительности
        monitor_thread = threading.Thread(target=self._monitor_performance)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Запускаем автоматическое переобучение
        retrain_thread = threading.Thread(target=self._auto_retrain_scheduler)
        retrain_thread.daemon = True
        retrain_thread.start()
        
        self.logger.info("🚀 Система реального времени запущена")
    
    def stop_real_time_system(self):
        """Остановить систему реального времени"""
        self.is_running = False
        self.logger.info("⏹️ Система реального времени остановлена")
    
    def _process_data_queue(self):
        """Обработка очереди данных"""
        while self.is_running:
            try:
                if not self.data_queue.empty():
                    match_data = self.data_queue.get(timeout=1)
                    self._process_match_data(match_data)
                else:
                    time.sleep(1)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Ошибка обработки данных: {e}")
    
    def _process_match_data(self, match_data: Dict):
        """Обработать данные матча"""
        try:
            # Извлекаем признаки
            features = self._extract_features(match_data)
            
            # Делаем прогноз
            prediction = self._make_prediction(features)
            
            # Сохраняем прогноз
            self._save_prediction(match_data, prediction)
            
            # Обновляем статистику
            self._update_prediction_stats(prediction)
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки матча: {e}")
    
    def _extract_features(self, match_data: Dict) -> np.ndarray:
        """Извлечь признаки из данных матча"""
        # Базовые признаки
        features = [
            match_data.get('home_team_form', 0.5),
            match_data.get('away_team_form', 0.5),
            match_data.get('home_goals_avg', 1.5),
            match_data.get('away_goals_avg', 1.5),
            match_data.get('home_conceded_avg', 1.2),
            match_data.get('away_conceded_avg', 1.2),
            match_data.get('home_advantage', 0.15),
            match_data.get('head_to_head_home', 0.5),
            match_data.get('weather_factor', 0.0),
            match_data.get('injury_factor', 0.0)
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _make_prediction(self, features: np.ndarray) -> Dict:
        """Сделать прогноз используя лучшую модель"""
        best_model = self._get_best_model()
        
        if best_model is None:
            return {'error': 'Нет обученных моделей'}
        
        try:
            # Получаем вероятности
            probabilities = best_model.predict_proba(features)[0]
            
            # Определяем результат
            result_classes = ['HOME_WIN', 'DRAW', 'AWAY_WIN']
            predicted_class = result_classes[np.argmax(probabilities)]
            confidence = np.max(probabilities)
            
            return {
                'predicted_result': predicted_class,
                'probabilities': {
                    'HOME_WIN': probabilities[0],
                    'DRAW': probabilities[1],
                    'AWAY_WIN': probabilities[2]
                },
                'confidence': confidence,
                'model_used': best_model.__class__.__name__,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования: {e}")
            return {'error': str(e)}
    
    def _get_best_model(self):
        """Получить лучшую модель на основе производительности"""
        if not self.model_performance:
            return None
        
        best_model_name = max(
            self.model_performance.keys(),
            key=lambda x: self.model_performance[x]['accuracy']
        )
        
        return self.models.get(best_model_name)
    
    def _save_prediction(self, match_data: Dict, prediction: Dict):
        """Сохранить прогноз"""
        prediction_record = {
            'match_id': match_data.get('match_id'),
            'home_team': match_data.get('home_team'),
            'away_team': match_data.get('away_team'),
            'prediction': prediction,
            'timestamp': datetime.now()
        }
        
        # Здесь можно сохранить в базу данных
        self.logger.info(f"Прогноз сохранен: {prediction_record['match_id']}")
    
    def _update_prediction_stats(self, prediction: Dict):
        """Обновить статистику прогнозов"""
        if 'error' in prediction:
            return
        
        # Обновляем счетчики
        for model_name in self.models.keys():
            self.model_performance[model_name]['predictions_count'] += 1
    
    def _monitor_performance(self):
        """Мониторинг производительности моделей"""
        while self.is_running:
            try:
                # Проверяем производительность каждые 10 минут
                time.sleep(600)
                
                for model_name, performance in self.model_performance.items():
                    if performance['predictions_count'] > 0:
                        accuracy = performance['correct_predictions'] / performance['predictions_count']
                        performance['accuracy'] = accuracy
                        
                        self.logger.info(f"Модель {model_name}: точность {accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Ошибка мониторинга: {e}")
    
    def _auto_retrain_scheduler(self):
        """Планировщик автоматического переобучения"""
        while self.is_running:
            try:
                # Проверяем необходимость переобучения каждые 6 часов
                time.sleep(21600)
                
                if self._should_retrain():
                    self.logger.info("🔄 Начинаем автоматическое переобучение")
                    self._retrain_models()
                
            except Exception as e:
                self.logger.error(f"Ошибка планировщика: {e}")
    
    def _should_retrain(self) -> bool:
        """Проверить, нужно ли переобучать модели"""
        for model_name, performance in self.model_performance.items():
            if performance['accuracy'] < 0.7:  # Порог точности
                return True
        
        return False
    
    def _retrain_models(self):
        """Переобучить модели"""
        try:
            # Загружаем новые данные
            new_data = self._load_training_data()
            
            if new_data is None or len(new_data) < 100:
                self.logger.warning("Недостаточно данных для переобучения")
                return
            
            # Переобучаем каждую модель
            for model_name, model in self.models.items():
                X, y = self._prepare_training_data(new_data)
                
                # Обучаем модель
                model.fit(X, y)
                
                # Оцениваем производительность
                cv_scores = cross_val_score(model, X, y, cv=5)
                accuracy = cv_scores.mean()
                
                # Обновляем производительность
                self.model_performance[model_name]['accuracy'] = accuracy
                self.model_performance[model_name]['last_update'] = datetime.now()
                
                self.logger.info(f"Модель {model_name} переобучена. Точность: {accuracy:.3f}")
            
            # Сохраняем модели
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Ошибка переобучения: {e}")
    
    def _load_training_data(self) -> Optional[pd.DataFrame]:
        """Загрузить данные для обучения"""
        # В реальном проекте здесь загрузка из базы данных
        # Для демонстрации создаем синтетические данные
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'home_team_form': np.random.uniform(0.2, 0.8, n_samples),
            'away_team_form': np.random.uniform(0.2, 0.8, n_samples),
            'home_goals_avg': np.random.uniform(1, 3, n_samples),
            'away_goals_avg': np.random.uniform(1, 3, n_samples),
            'home_conceded_avg': np.random.uniform(0.5, 2, n_samples),
            'away_conceded_avg': np.random.uniform(0.5, 2, n_samples),
            'home_advantage': np.random.uniform(0.1, 0.2, n_samples),
            'head_to_head_home': np.random.uniform(0.3, 0.7, n_samples),
            'weather_factor': np.random.uniform(-0.1, 0.1, n_samples),
            'injury_factor': np.random.uniform(-0.2, 0.1, n_samples)
        }
        
        # Создаем результаты с учетом домашнего преимущества
        results = []
        for i in range(n_samples):
            home_advantage = data['home_advantage'][i]
            form_diff = data['home_team_form'][i] - data['away_team_form'][i]
            
            home_win_prob = 0.33 + home_advantage + form_diff * 0.1
            draw_prob = 0.25
            away_win_prob = 1 - home_win_prob - draw_prob
            
            results.append(np.random.choice(
                ['HOME_WIN', 'DRAW', 'AWAY_WIN'],
                p=[home_win_prob, draw_prob, away_win_prob]
            ))
        
        data['result'] = results
        return pd.DataFrame(data)
    
    def _prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Подготовить данные для обучения"""
        feature_columns = [
            'home_team_form', 'away_team_form', 'home_goals_avg',
            'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg',
            'home_advantage', 'head_to_head_home', 'weather_factor', 'injury_factor'
        ]
        
        X = df[feature_columns].values
        y = df['result'].values
        
        return X, y
    
    def _save_models(self):
        """Сохранить модели"""
        for model_name, model in self.models.items():
            filename = f"models/{model_name}_model.pkl"
            joblib.dump(model, filename)
            self.logger.info(f"Модель {model_name} сохранена в {filename}")
    
    def get_system_status(self) -> Dict:
        """Получить статус системы"""
        return {
            'is_running': self.is_running,
            'queue_size': self.data_queue.qsize(),
            'model_performance': self.model_performance,
            'last_update': datetime.now().isoformat()
        }
    
    def update_model_feedback(self, match_id: str, actual_result: str, predicted_result: str):
        """Обновить обратную связь о точности прогноза"""
        is_correct = actual_result == predicted_result
        
        for model_name in self.models.keys():
            if is_correct:
                self.model_performance[model_name]['correct_predictions'] += 1
        
        self.logger.info(f"Обратная связь для матча {match_id}: {'правильно' if is_correct else 'неправильно'}")

# Пример использования
if __name__ == "__main__":
    # Создаем систему реального времени
    rt_system = RealTimeMLSystem()
    
    print("🚀 СИСТЕМА МАШИННОГО ОБУЧЕНИЯ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 60)
    
    # Запускаем систему
    rt_system.start_real_time_system()
    
    # Симулируем добавление данных
    print("\n📊 Симуляция добавления данных...")
    
    for i in range(5):
        match_data = {
            'match_id': f'match_{i+1}',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'home_team_form': np.random.uniform(0.4, 0.8),
            'away_team_form': np.random.uniform(0.4, 0.8),
            'home_goals_avg': np.random.uniform(1.2, 2.5),
            'away_goals_avg': np.random.uniform(1.2, 2.5),
            'home_conceded_avg': np.random.uniform(0.8, 1.8),
            'away_conceded_avg': np.random.uniform(0.8, 1.8),
            'home_advantage': 0.15,
            'head_to_head_home': np.random.uniform(0.3, 0.7),
            'weather_factor': np.random.uniform(-0.1, 0.1),
            'injury_factor': np.random.uniform(-0.2, 0.1)
        }
        
        rt_system.add_new_data(match_data)
        time.sleep(1)
    
    # Показываем статус системы
    time.sleep(2)
    status = rt_system.get_system_status()
    
    print(f"\n📈 СТАТУС СИСТЕМЫ:")
    print(f"Запущена: {status['is_running']}")
    print(f"Размер очереди: {status['queue_size']}")
    print(f"Последнее обновление: {status['last_update']}")
    
    # Останавливаем систему
    time.sleep(5)
    rt_system.stop_real_time_system()
    
    print("\n✅ Система остановлена")
