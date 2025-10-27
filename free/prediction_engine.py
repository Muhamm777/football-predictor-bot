import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FootballPredictionEngine:
    """Движок для прогнозирования результатов футбольных матчей"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовить признаки для модели"""
        # Создаем копию датафрейма
        features_df = df.copy()
        
        # Кодируем категориальные переменные
        categorical_columns = ['home_team', 'away_team', 'competition']
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    # Для новых данных используем существующий энкодер
                    try:
                        features_df[f'{col}_encoded'] = self.label_encoders[col].transform(features_df[col].astype(str))
                    except ValueError:
                        # Если встречается новое значение, добавляем его
                        features_df[col] = features_df[col].astype(str)
                        features_df[f'{col}_encoded'] = 0  # Значение по умолчанию
        
        # Создаем дополнительные признаки
        features_df = self._create_additional_features(features_df)
        
        # Выбираем числовые признаки для модели
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_columns if col != 'result_encoded']
        
        return features_df
    
    def _create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создать дополнительные признаки для улучшения прогнозов"""
        # Статистика за последние N матчей
        df = self._add_recent_form_features(df)
        
        # Домашнее преимущество
        df['is_home_advantage'] = 1  # Всегда 1 для домашней команды
        
        # День недели матча
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'])
            df['day_of_week'] = df['match_date'].dt.dayofweek
            df['month'] = df['match_date'].dt.month
        
        return df
    
    def _add_recent_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить признаки формы команд за последние матчи"""
        # Это упрощенная версия - в реальном проекте нужно получать исторические данные
        # Создаем случайные значения для демонстрации
        np.random.seed(42)
        
        df['home_team_form'] = np.random.uniform(0.2, 0.8, len(df))
        df['away_team_form'] = np.random.uniform(0.2, 0.8, len(df))
        df['home_team_goals_avg'] = np.random.uniform(1, 3, len(df))
        df['away_team_goals_avg'] = np.random.uniform(1, 3, len(df))
        df['home_team_conceded_avg'] = np.random.uniform(0.5, 2, len(df))
        df['away_team_conceded_avg'] = np.random.uniform(0.5, 2, len(df))
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """Обучить модель на исторических данных"""
        print("Подготовка данных для обучения...")
        
        # Подготавливаем признаки
        features_df = self.prepare_features(df)
        
        # Кодируем целевую переменную
        if 'result' in features_df.columns:
            result_encoder = LabelEncoder()
            features_df['result_encoded'] = result_encoder.fit_transform(features_df['result'])
            self.label_encoders['result'] = result_encoder
        
        # Разделяем на признаки и целевую переменную
        X = features_df[self.feature_columns].fillna(0)
        y = features_df['result_encoded']
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Нормализуем признаки
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Создаем и обучаем модель (базовая)
        base = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        print("Обучение модели...")
        base.fit(X_train_scaled, y_train)
        # Калибровка вероятностей (sigmoid для скорости/стабильности)
        try:
            self.model = CalibratedClassifierCV(base, method='sigmoid', cv=3)
            self.model.fit(X_train_scaled, y_train)
        except Exception:
            # fallback: используем базовую модель без калибровки
            self.model = base
        
        # Оцениваем качество модели
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        # Метрики вероятностей
        try:
            y_proba = self.model.predict_proba(X_test_scaled)
            # logloss (multiclass)
            ll = log_loss(y_test, y_proba, labels=np.unique(y_test))
            # brier (multiclass): среднее по классам (one-vs-all)
            # строим one-hot для истинных меток
            classes_ = np.unique(y_test)
            k = len(classes_)
            oh = np.zeros((y_test.shape[0], k), dtype=float)
            # сопоставим позиции классов
            class_to_idx = {c: i for i, c in enumerate(classes_)}
            for i, yy in enumerate(y_test):
                oh[i, class_to_idx.get(yy, 0)] = 1.0
            br = float(np.mean(np.sum((y_proba[:, :k] - oh) ** 2, axis=1)))
        except Exception:
            ll = None
            br = None
        
        print(f"Точность модели: {accuracy:.3f}")
        print("\nОтчет по классификации:")
        print(classification_report(y_test, y_pred, target_names=result_encoder.classes_))
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'logloss': ll,
            'brier': br,
            'feature_importance': dict(zip(self.feature_columns, getattr(getattr(self.model, 'base_estimator', None) or base, 'feature_importances_', np.zeros(len(self.feature_columns))))),
            'model_type': 'Calibrated(GB, sigmoid)' if isinstance(self.model, CalibratedClassifierCV) else 'GradientBoostingClassifier'
        }
    
    def predict_match(self, home_team: str, away_team: str, competition: str = "Premier League") -> Dict:
        """Спрогнозировать результат матча"""
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train_model()")
        
        # Создаем данные для предсказания
        match_data = {
            'home_team': [home_team],
            'away_team': [away_team],
            'competition': [competition],
            'match_date': [pd.Timestamp.now()]
        }
        
        df = pd.DataFrame(match_data)
        features_df = self.prepare_features(df)
        
        # Подготавливаем признаки
        X = features_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Делаем предсказание
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        prediction_encoded = self.model.predict(X_scaled)[0]
        
        # Декодируем результат через label encoder
        result_encoder = self.label_encoders.get('result')
        result_classes = result_encoder.classes_ if result_encoder is not None else None
        if result_encoder is not None:
            try:
                predicted_result = result_encoder.inverse_transform([prediction_encoded])[0]
            except Exception:
                # fallback на индексацию по classes_
                predicted_result = (result_classes[int(prediction_encoded)]
                                    if result_classes is not None and 0 <= int(prediction_encoded) < len(result_classes)
                                    else 'HOME_WIN')
        else:
            # На всякий случай, если энкодер отсутствует
            predicted_result = 'HOME_WIN'
        
        # Создаем детальный прогноз, маппя вероятности по именам классов
        if result_classes is not None:
            by_class = {cls: float(prediction_proba[i]) for i, cls in enumerate(result_classes)}
            probabilities = {
                'HOME_WIN': by_class.get('HOME_WIN', 0.0),
                'DRAW': by_class.get('DRAW', 0.0),
                'AWAY_WIN': by_class.get('AWAY_WIN', 0.0),
            }
        else:
            # Fallback: сохраняем исходный порядок, если классы неизвестны
            probabilities = {
                'HOME_WIN': float(prediction_proba[0]) if len(prediction_proba) > 0 else 0.0,
                'DRAW': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.0,
                'AWAY_WIN': float(prediction_proba[2]) if len(prediction_proba) > 2 else 0.0,
            }
        
        confidence = max(probabilities.values())
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_result': predicted_result,
            'probabilities': probabilities,
            'confidence': confidence,
            'recommendation': self._get_recommendation(predicted_result, confidence)
        }
    
    def _get_recommendation(self, result: str, confidence: float) -> str:
        """Получить рекомендацию на основе прогноза"""
        if confidence < 0.4:
            return "Низкая уверенность - не рекомендуется ставить"
        elif confidence < 0.6:
            return "Средняя уверенность - осторожная ставка"
        elif confidence < 0.8:
            return "Высокая уверенность - рекомендуемая ставка"
        else:
            return "Очень высокая уверенность - сильная рекомендация"
    
    def save_model(self, filepath: str = 'football_model.pkl'):
        """Сохранить обученную модель"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str = 'football_model.pkl'):
        """Загрузить сохраненную модель"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        print(f"Модель загружена из {filepath}")

# Пример использования
if __name__ == "__main__":
    # Создаем тестовые данные
    np.random.seed(42)
    n_matches = 1000
    
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 
             'Tottenham', 'Barcelona', 'Real Madrid', 'Atletico Madrid', 'Bayern Munich']
    
    test_data = {
        'home_team': np.random.choice(teams, n_matches),
        'away_team': np.random.choice(teams, n_matches),
        'competition': np.random.choice(['Premier League', 'La Liga', 'Bundesliga'], n_matches),
        'match_date': pd.date_range('2023-01-01', periods=n_matches, freq='D')
    }
    
    # Создаем случайные результаты
    results = np.random.choice(['HOME_WIN', 'DRAW', 'AWAY_WIN'], n_matches, p=[0.45, 0.25, 0.30])
    test_data['result'] = results
    
    df = pd.DataFrame(test_data)
    
    # Создаем и обучаем модель
    engine = FootballPredictionEngine()
    metrics = engine.train_model(df)
    
    # Тестируем прогноз
    prediction = engine.predict_match('Arsenal', 'Chelsea', 'Premier League')
    print(f"\nПрогноз: {prediction}")
    
    # Сохраняем модель
    engine.save_model()
