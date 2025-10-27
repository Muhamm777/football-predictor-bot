"""
Детальное описание факторов, на основе которых бот делает прогнозы
"""

class PredictionFactors:
    """Факторы, влияющие на прогнозы футбольного бота"""
    
    def __init__(self):
        self.factors = {
            # 1. ИСТОРИЧЕСКИЕ ДАННЫЕ (40% влияния)
            "historical_data": {
                "head_to_head": "Результаты личных встреч команд",
                "home_advantage": "Домашнее преимущество (в среднем 60% побед дома)",
                "season_form": "Форма команды в текущем сезоне",
                "recent_matches": "Результаты последних 5-10 матчей"
            },
            
            # 2. СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ (30% влияния)
            "statistical_indicators": {
                "goals_scored_avg": "Среднее количество голов за матч",
                "goals_conceded_avg": "Среднее количество пропущенных голов",
                "possession_percentage": "Процент владения мячом",
                "shots_on_target": "Удары в створ ворот",
                "pass_accuracy": "Точность передач",
                "defensive_strength": "Надежность защиты"
            },
            
            # 3. ФОРМА КОМАНД (20% влияния)
            "team_form": {
                "win_streak": "Серия побед/поражений",
                "goals_in_form": "Голы в последних матчах",
                "clean_sheets": "Сухие матчи",
                "injury_impact": "Влияние травм ключевых игроков",
                "suspensions": "Дисквалификации игроков"
            },
            
            # 4. ВНЕШНИЕ ФАКТОРЫ (10% влияния)
            "external_factors": {
                "weather_conditions": "Погодные условия",
                "stadium_factor": "Фактор стадиона",
                "crowd_support": "Поддержка болельщиков",
                "fixture_congestion": "Плотность календаря",
                "motivation_level": "Мотивация команд"
            }
        }
    
    def get_prediction_algorithm(self):
        """Алгоритм принятия решений бота"""
        return {
            "step_1": "Сбор и анализ исторических данных матчей",
            "step_2": "Вычисление статистических показателей команд",
            "step_3": "Анализ текущей формы команд",
            "step_4": "Учет внешних факторов",
            "step_5": "Применение алгоритма машинного обучения",
            "step_6": "Вычисление вероятностей исходов",
            "step_7": "Формирование финального прогноза"
        }
    
    def get_feature_importance(self):
        """Важность различных признаков для прогноза"""
        return {
            "home_advantage": 0.15,  # 15% влияния
            "recent_form": 0.20,     # 20% влияния
            "head_to_head": 0.10,    # 10% влияния
            "goals_statistics": 0.25, # 25% влияния
            "team_strength": 0.15,   # 15% влияния
            "external_factors": 0.10, # 10% влияния
            "random_factor": 0.05    # 5% случайности
        }

# Примеры конкретных данных, которые анализирует бот
class DataExamples:
    """Примеры данных, на основе которых бот делает прогнозы"""
    
    @staticmethod
    def get_team_analysis_example():
        """Пример анализа команды"""
        return {
            "team": "Arsenal",
            "recent_form": {
                "last_5_matches": ["W", "W", "D", "W", "L"],
                "goals_scored": 8,
                "goals_conceded": 3,
                "clean_sheets": 3
            },
            "home_record": {
                "matches_played": 10,
                "wins": 7,
                "draws": 2,
                "losses": 1,
                "goals_scored": 18,
                "goals_conceded": 6
            },
            "head_to_head": {
                "opponent": "Chelsea",
                "last_5_meetings": ["L", "W", "D", "W", "L"],
                "home_advantage": True
            }
        }
    
    @staticmethod
    def get_prediction_calculation():
        """Пример расчета прогноза"""
        return {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "factors_analysis": {
                "home_advantage": 0.15,  # +15% к победе дома
                "recent_form_arsenal": 0.20,  # +20% (хорошая форма)
                "recent_form_chelsea": -0.10,  # -10% (плохая форма)
                "head_to_head": 0.05,  # +5% (историческое преимущество)
                "goals_statistics": 0.10,  # +10% (больше голов забивают)
                "injuries": -0.05,  # -5% (травмы ключевых игроков)
            },
            "final_probabilities": {
                "home_win": 0.55,  # 55% вероятность победы дома
                "draw": 0.25,      # 25% вероятность ничьи
                "away_win": 0.20   # 20% вероятность победы в гостях
            },
            "confidence": 0.75,  # 75% уверенность в прогнозе
            "recommendation": "Рекомендуется ставка на победу Arsenal"
        }

# Математическая модель прогнозирования
class PredictionModel:
    """Математическая модель для расчета прогнозов"""
    
    def __init__(self):
        self.weights = {
            'home_advantage': 0.15,
            'form_difference': 0.25,
            'goals_difference': 0.20,
            'head_to_head': 0.10,
            'strength_rating': 0.20,
            'external_factors': 0.10
        }
    
    def calculate_prediction(self, match_data):
        """Расчет прогноза на основе данных матча"""
        # Пример расчета
        home_advantage = 0.15  # Базовое домашнее преимущество
        form_factor = self._calculate_form_factor(match_data)
        goals_factor = self._calculate_goals_factor(match_data)
        h2h_factor = self._calculate_h2h_factor(match_data)
        
        # Итоговая вероятность победы дома
        home_win_probability = 0.33 + home_advantage + form_factor + goals_factor + h2h_factor
        
        # Нормализация вероятностей
        total = home_win_probability + 0.25 + (1 - home_win_probability - 0.25)
        
        return {
            'home_win': home_win_probability / total,
            'draw': 0.25 / total,
            'away_win': (1 - home_win_probability - 0.25) / total
        }
    
    def _calculate_form_factor(self, match_data):
        """Расчет фактора формы команд"""
        home_form = match_data['home_team']['recent_form_score']
        away_form = match_data['away_team']['recent_form_score']
        return (home_form - away_form) * 0.1
    
    def _calculate_goals_factor(self, match_data):
        """Расчет фактора голов"""
        home_goals_avg = match_data['home_team']['goals_scored_avg']
        away_goals_avg = match_data['away_team']['goals_scored_avg']
        home_conceded_avg = match_data['home_team']['goals_conceded_avg']
        away_conceded_avg = match_data['away_team']['goals_conceded_avg']
        
        attack_difference = (home_goals_avg - away_goals_avg) * 0.05
        defense_difference = (away_conceded_avg - home_conceded_avg) * 0.05
        
        return attack_difference + defense_difference
    
    def _calculate_h2h_factor(self, match_data):
        """Расчет фактора личных встреч"""
        h2h_record = match_data['head_to_head']
        home_wins = h2h_record['home_wins']
        total_matches = h2h_record['total_matches']
        
        if total_matches > 0:
            return (home_wins / total_matches - 0.5) * 0.1
        return 0

if __name__ == "__main__":
    # Демонстрация факторов прогнозирования
    factors = PredictionFactors()
    print("🔍 Факторы, влияющие на прогнозы бота:")
    
    for category, items in factors.factors.items():
        print(f"\n📊 {category.upper()}:")
        for key, description in items.items():
            print(f"   • {key}: {description}")
    
    print("\n🧮 Алгоритм принятия решений:")
    for step, description in factors.get_prediction_algorithm().items():
        print(f"   {step}: {description}")
    
    print("\n⚖️ Важность признаков:")
    for feature, importance in factors.get_feature_importance().items():
        print(f"   {feature}: {importance*100:.1f}%")
