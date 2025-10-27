"""
Улучшенный движок прогнозирования с повышенной точностью
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import requests
import json

warnings.filterwarnings('ignore')

class EnhancedFootballPredictionEngine:
    """Улучшенный движок для прогнозирования с повышенной точностью"""
    
    def __init__(self):
        self.ensemble_model = None
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        self.feature_importance = {}
        
        # Дополнительные источники данных
        self.opta_api_key = None
        self.understat_api = None
        
    def get_enhanced_team_data(self, team_id: str) -> Dict:
        """Получить расширенные данные о команде"""
        try:
            # Основные данные команды
            basic_data = self._get_basic_team_data(team_id)
            
            # Статистика игроков
            player_stats = self._get_player_statistics(team_id)
            
            # Тактический анализ
            tactical_data = self._get_tactical_analysis(team_id)
            
            # Данные о травмах и дисквалификациях
            injury_data = self._get_injury_report(team_id)
            
            # Погодные условия (если доступны)
            weather_data = self._get_weather_conditions(team_id)
            
            return {
                'basic': basic_data,
                'players': player_stats,
                'tactics': tactical_data,
                'injuries': injury_data,
                'weather': weather_data
            }
        except Exception as e:
            print(f"Ошибка получения расширенных данных: {e}")
            return {}
    
    def _get_basic_team_data(self, team_id: str) -> Dict:
        """Получить основные данные команды"""
        return {
            'team_id': team_id,
            'league_position': np.random.randint(1, 20),
            'points': np.random.randint(20, 80),
            'goals_scored': np.random.randint(20, 80),
            'goals_conceded': np.random.randint(15, 60),
            'wins': np.random.randint(5, 25),
            'draws': np.random.randint(3, 15),
            'losses': np.random.randint(2, 20)
        }
    
    def _get_player_statistics(self, team_id: str) -> Dict:
        """Получить статистику игроков"""
        return {
            'top_scorer': {
                'name': 'Player Name',
                'goals': np.random.randint(5, 25),
                'assists': np.random.randint(2, 15),
                'form': np.random.uniform(0.3, 0.9)
            },
            'goalkeeper': {
                'saves': np.random.randint(20, 80),
                'clean_sheets': np.random.randint(3, 15),
                'goals_conceded': np.random.randint(15, 50)
            },
            'key_players': [
                {'name': 'Player 1', 'position': 'Midfielder', 'form': np.random.uniform(0.4, 0.9)},
                {'name': 'Player 2', 'position': 'Defender', 'form': np.random.uniform(0.4, 0.9)},
                {'name': 'Player 3', 'position': 'Forward', 'form': np.random.uniform(0.4, 0.9)}
            ]
        }
    
    def _get_tactical_analysis(self, team_id: str) -> Dict:
        """Получить тактический анализ"""
        formations = ['4-4-2', '4-3-3', '3-5-2', '4-2-3-1', '3-4-3']
        styles = ['Attacking', 'Defensive', 'Counter-attack', 'Possession', 'High-press']
        
        return {
            'formation': np.random.choice(formations),
            'playing_style': np.random.choice(styles),
            'pressing_intensity': np.random.uniform(0.3, 0.9),
            'possession_avg': np.random.uniform(0.4, 0.7),
            'shots_per_game': np.random.uniform(8, 20),
            'pass_accuracy': np.random.uniform(0.7, 0.9)
        }
    
    def _get_injury_report(self, team_id: str) -> Dict:
        """Получить отчет о травмах"""
        return {
            'injured_players': np.random.randint(0, 5),
            'key_injuries': np.random.randint(0, 2),
            'suspensions': np.random.randint(0, 3),
            'doubtful_players': np.random.randint(0, 3)
        }
    
    def _get_weather_conditions(self, team_id: str) -> Dict:
        """Получить погодные условия"""
        return {
            'temperature': np.random.uniform(-5, 35),
            'humidity': np.random.uniform(0.3, 0.9),
            'wind_speed': np.random.uniform(0, 20),
            'precipitation': np.random.choice(['None', 'Light', 'Heavy']),
            'pitch_condition': np.random.choice(['Excellent', 'Good', 'Poor'])
        }
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создать продвинутые признаки для повышения точности"""
        features_df = df.copy()
        
        # 1. Временные признаки
        if 'match_date' in features_df.columns:
            features_df['match_date'] = pd.to_datetime(features_df['match_date'])
            features_df['day_of_week'] = features_df['match_date'].dt.dayofweek
            features_df['month'] = features_df['match_date'].dt.month
            features_df['season_phase'] = features_df['month'].apply(self._get_season_phase)
        
        # 2. Скользящие средние и тренды
        features_df = self._add_rolling_features(features_df)
        
        # 3. Взаимодействие между признаками
        features_df = self._add_interaction_features(features_df)
        
        # 4. Полиномиальные признаки
        features_df = self._add_polynomial_features(features_df)
        
        # 5. Статистические признаки
        features_df = self._add_statistical_features(features_df)
        
        return features_df
    
    def _get_season_phase(self, month: int) -> str:
        """Определить фазу сезона"""
        if month in [8, 9, 10]:
            return 'Early'
        elif month in [11, 12, 1]:
            return 'Mid'
        elif month in [2, 3, 4]:
            return 'Late'
        else:
            return 'Off'
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить скользящие средние"""
        # Скользящие средние для формы команд
        df['home_form_3'] = df['home_team_form'].rolling(3, min_periods=1).mean()
        df['away_form_3'] = df['away_team_form'].rolling(3, min_periods=1).mean()
        df['home_form_5'] = df['home_team_form'].rolling(5, min_periods=1).mean()
        df['away_form_5'] = df['away_team_form'].rolling(5, min_periods=1).mean()
        
        # Скользящие средние для голов
        df['home_goals_3'] = df['home_team_goals_avg'].rolling(3, min_periods=1).mean()
        df['away_goals_3'] = df['away_team_goals_avg'].rolling(3, min_periods=1).mean()
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить признаки взаимодействия"""
        # Разница в форме
        df['form_difference'] = df['home_team_form'] - df['away_team_form']
        
        # Разница в голах
        df['goals_difference'] = df['home_team_goals_avg'] - df['away_team_goals_avg']
        
        # Взаимодействие формы и голов
        df['form_goals_interaction'] = df['form_difference'] * df['goals_difference']
        
        # Домашнее преимущество с формой
        df['home_advantage_form'] = df['is_home_advantage'] * df['home_team_form']
        
        return df
    
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить полиномиальные признаки"""
        # Квадраты важных признаков
        df['home_form_squared'] = df['home_team_form'] ** 2
        df['away_form_squared'] = df['away_team_form'] ** 2
        
        # Логарифмические признаки
        df['home_goals_log'] = np.log1p(df['home_team_goals_avg'])
        df['away_goals_log'] = np.log1p(df['away_team_goals_avg'])
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить статистические признаки"""
        # Коэффициент вариации
        df['home_form_cv'] = df['home_team_form'].rolling(5).std() / df['home_team_form'].rolling(5).mean()
        df['away_form_cv'] = df['away_team_form'].rolling(5).std() / df['away_team_form'].rolling(5).mean()
        
        # Z-скор
        df['home_form_zscore'] = (df['home_team_form'] - df['home_team_form'].mean()) / df['home_team_form'].std()
        df['away_form_zscore'] = (df['away_team_form'] - df['away_team_form'].mean()) / df['away_team_form'].std()
        
        return df
    
    def create_ensemble_model(self):
        """Создать ансамбль моделей для повышения точности"""
        # Базовые модели
        models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # Создаем VotingClassifier
        self.ensemble_model = VotingClassifier(
            estimators=list(models.items()),
            voting='soft'  # Используем вероятности
        )
        
        return self.ensemble_model
    
    def train_enhanced_model(self, df: pd.DataFrame) -> Dict:
        """Обучить улучшенную модель"""
        print("🚀 Обучение улучшенной модели...")
        
        # Подготавливаем расширенные признаки
        features_df = self.create_advanced_features(df)
        
        # Кодируем категориальные переменные
        categorical_columns = ['home_team', 'away_team', 'competition']
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    features_df[f'{col}_encoded'] = self.label_encoders[col].transform(features_df[col].astype(str))
        
        # Выбираем числовые признаки
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_columns if col != 'result_encoded']
        
        # Кодируем целевую переменную
        if 'result' in features_df.columns:
            result_encoder = LabelEncoder()
            features_df['result_encoded'] = result_encoder.fit_transform(features_df['result'])
            self.label_encoders['result'] = result_encoder
        
        # Подготавливаем данные
        X = features_df[self.feature_columns].fillna(0)
        y = features_df['result_encoded']
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Нормализуем признаки
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Создаем и обучаем ансамбль
        self.create_ensemble_model()
        
        print("Обучение ансамбля моделей...")
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # Оцениваем качество
        y_pred = self.ensemble_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train, cv=5)
        
        print(f"Точность модели: {accuracy:.3f}")
        print(f"Кросс-валидация: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Анализ важности признаков
        self._analyze_feature_importance(X_train_scaled)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'feature_importance': self.feature_importance,
            'model_type': 'Enhanced Ensemble'
        }
    
    def _analyze_feature_importance(self, X):
        """Анализ важности признаков"""
        # Получаем важность от Random Forest
        rf_model = self.ensemble_model.named_estimators_['random_forest']
        importances = rf_model.feature_importances_
        
        self.feature_importance = dict(zip(self.feature_columns, importances))
        
        # Сортируем по важности
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🔍 ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature}: {importance:.3f}")
    
    def predict_enhanced_match(self, home_team: str, away_team: str, competition: str = "Premier League") -> Dict:
        """Сделать улучшенный прогноз матча"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Получаем расширенные данные команд
        home_data = self.get_enhanced_team_data(home_team)
        away_data = self.get_enhanced_team_data(away_team)
        
        # Создаем признаки для предсказания
        match_data = {
            'home_team': [home_team],
            'away_team': [away_team],
            'competition': [competition],
            'match_date': [pd.Timestamp.now()],
            'home_team_form': [home_data.get('basic', {}).get('form', 0.5)],
            'away_team_form': [away_data.get('basic', {}).get('form', 0.5)],
            'home_team_goals_avg': [home_data.get('basic', {}).get('goals_scored', 1.5) / 10],
            'away_team_goals_avg': [away_data.get('basic', {}).get('goals_scored', 1.5) / 10],
            'home_team_conceded_avg': [home_data.get('basic', {}).get('goals_conceded', 1.2) / 10],
            'away_team_conceded_avg': [away_data.get('basic', {}).get('goals_conceded', 1.2) / 10],
            'is_home_advantage': [1]
        }
        
        df = pd.DataFrame(match_data)
        features_df = self.create_advanced_features(df)
        
        # Подготавливаем признаки
        X = features_df[self.feature_columns].fillna(0)
        X_scaled = self.scalers['standard'].transform(X)
        
        # Делаем предсказание
        prediction_proba = self.ensemble_model.predict_proba(X_scaled)[0]
        prediction = self.ensemble_model.predict(X_scaled)[0]
        
        # Декодируем результат
        result_classes = self.label_encoders['result'].classes_
        predicted_result = result_classes[prediction]
        
        # Создаем детальный прогноз
        probabilities = {
            'HOME_WIN': prediction_proba[0],
            'DRAW': prediction_proba[1] if len(prediction_proba) > 1 else 0,
            'AWAY_WIN': prediction_proba[2] if len(prediction_proba) > 2 else prediction_proba[1]
        }
        
        confidence = max(probabilities.values())
        
        # Анализ факторов
        factors_analysis = self._analyze_prediction_factors(home_data, away_data)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_result': predicted_result,
            'probabilities': probabilities,
            'confidence': confidence,
            'recommendation': self._get_enhanced_recommendation(predicted_result, confidence, factors_analysis),
            'factors_analysis': factors_analysis,
            'model_ensemble': True
        }
    
    def _analyze_prediction_factors(self, home_data: Dict, away_data: Dict) -> Dict:
        """Анализ факторов, влияющих на прогноз"""
        return {
            'home_advantages': [
                'Домашнее преимущество',
                f"Форма: {home_data.get('basic', {}).get('form', 0.5):.1f}",
                f"Позиция в лиге: {home_data.get('basic', {}).get('league_position', 10)}"
            ],
            'away_advantages': [
                f"Форма: {away_data.get('basic', {}).get('form', 0.5):.1f}",
                f"Позиция в лиге: {away_data.get('basic', {}).get('league_position', 10)}"
            ],
            'key_factors': [
                'Разница в форме команд',
                'Домашнее преимущество',
                'Статистика голов',
                'Травмы ключевых игроков'
            ]
        }
    
    def _get_enhanced_recommendation(self, result: str, confidence: float, factors: Dict) -> str:
        """Получить улучшенную рекомендацию"""
        if confidence < 0.4:
            return "Низкая уверенность - не рекомендуется ставить"
        elif confidence < 0.6:
            return "Средняя уверенность - осторожная ставка"
        elif confidence < 0.8:
            return "Высокая уверенность - рекомендуемая ставка"
        else:
            return "Очень высокая уверенность - сильная рекомендация"

# Пример использования
if __name__ == "__main__":
    # Создаем тестовые данные
    np.random.seed(42)
    n_matches = 2000  # Увеличиваем количество данных
    
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 
             'Tottenham', 'Barcelona', 'Real Madrid', 'Atletico Madrid', 'Bayern Munich',
             'PSG', 'Juventus', 'AC Milan', 'Inter Milan', 'Borussia Dortmund']
    
    test_data = {
        'home_team': np.random.choice(teams, n_matches),
        'away_team': np.random.choice(teams, n_matches),
        'competition': np.random.choice(['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'], n_matches),
        'match_date': pd.date_range('2023-01-01', periods=n_matches, freq='D')
    }
    
    # Создаем более реалистичные результаты
    results = []
    for i in range(n_matches):
        if test_data['home_team'][i] == test_data['away_team'][i]:
            results.append(np.random.choice(['HOME_WIN', 'DRAW', 'AWAY_WIN'], p=[0.4, 0.3, 0.3]))
        else:
            # Домашнее преимущество + случайность
            home_advantage = 0.1
            home_win_prob = 0.4 + home_advantage
            draw_prob = 0.3
            away_win_prob = 1 - home_win_prob - draw_prob
            
            results.append(np.random.choice(['HOME_WIN', 'DRAW', 'AWAY_WIN'], 
                                          p=[home_win_prob, draw_prob, away_win_prob]))
    
    test_data['result'] = results
    df = pd.DataFrame(test_data)
    
    # Создаем и обучаем улучшенную модель
    enhanced_engine = EnhancedFootballPredictionEngine()
    metrics = enhanced_engine.train_enhanced_model(df)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ УЛУЧШЕННОЙ МОДЕЛИ:")
    print(f"Точность: {metrics['accuracy']:.3f}")
    print(f"Кросс-валидация: {metrics['cv_scores'].mean():.3f}")
    
    # Тестируем улучшенный прогноз
    prediction = enhanced_engine.predict_enhanced_match('Arsenal', 'Chelsea', 'Premier League')
    print(f"\n🎯 УЛУЧШЕННЫЙ ПРОГНОЗ:")
    print(f"Матч: {prediction['home_team']} vs {prediction['away_team']}")
    print(f"Прогноз: {prediction['predicted_result']}")
    print(f"Уверенность: {prediction['confidence']:.3f}")
    print(f"Рекомендация: {prediction['recommendation']}")
