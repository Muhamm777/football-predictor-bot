import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from typing import List, Dict, Optional

class FootballDataCollector:
    """Класс для сбора данных о футбольных матчах"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {
            'X-Auth-Token': api_key,
            'Content-Type': 'application/json'
        } if api_key else {}
        
    def get_competitions(self) -> List[Dict]:
        """Получить список доступных лиг"""
        try:
            response = requests.get(f"{self.base_url}/competitions", headers=self.headers)
            response.raise_for_status()
            return response.json()['competitions']
        except Exception as e:
            print(f"Ошибка получения лиг: {e}")
            return []
    
    def get_matches(self, competition_id: str, date_from: str = None, date_to: str = None) -> List[Dict]:
        """Получить матчи из определенной лиги"""
        try:
            url = f"{self.base_url}/competitions/{competition_id}/matches"
            params = {}
            
            if date_from:
                params['dateFrom'] = date_from
            if date_to:
                params['dateTo'] = date_to
                
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()['matches']
        except Exception as e:
            print(f"Ошибка получения матчей: {e}")
            return []
    
    def get_team_statistics(self, team_id: str) -> Dict:
        """Получить статистику команды"""
        try:
            response = requests.get(f"{self.base_url}/teams/{team_id}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ошибка получения статистики команды: {e}")
            return {}
    
    def get_match_statistics(self, match_id: str) -> Dict:
        """Получить детальную статистику матча"""
        try:
            response = requests.get(f"{self.base_url}/matches/{match_id}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ошибка получения статистики матча: {e}")
            return {}
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Dict]:
        """Получить предстоящие матчи на указанное количество дней"""
        today = datetime.now().strftime('%Y-%m-%d')
        future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        # Получаем матчи из популярных лиг
        popular_leagues = ['PL', 'PD', 'SA', 'BL1', 'FL1', 'CL']  # Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League
        
        all_matches = []
        for league in popular_leagues:
            matches = self.get_matches(league, today, future_date)
            all_matches.extend(matches)
            time.sleep(0.1)  # Задержка между запросами
            
        return all_matches
    
    def process_match_data(self, match: Dict) -> Dict:
        """Обработать данные матча для анализа"""
        processed = {
            'match_id': match.get('id'),
            'home_team': match.get('homeTeam', {}).get('name'),
            'away_team': match.get('awayTeam', {}).get('name'),
            'home_team_id': match.get('homeTeam', {}).get('id'),
            'away_team_id': match.get('awayTeam', {}).get('id'),
            'match_date': match.get('utcDate'),
            'status': match.get('status'),
            'competition': match.get('competition', {}).get('name'),
            'competition_id': match.get('competition', {}).get('id')
        }
        
        # Добавляем результаты, если матч завершен
        if match.get('status') == 'FINISHED':
            score = match.get('score', {}).get('fullTime', {})
            processed.update({
                'home_score': score.get('home'),
                'away_score': score.get('away'),
                'result': self._determine_result(score.get('home'), score.get('away'))
            })
        
        return processed
    
    def _determine_result(self, home_score: int, away_score: int) -> str:
        """Определить результат матча"""
        if home_score > away_score:
            return 'HOME_WIN'
        elif away_score > home_score:
            return 'AWAY_WIN'
        else:
            return 'DRAW'
    
    def save_matches_to_csv(self, matches: List[Dict], filename: str = 'matches.csv'):
        """Сохранить матчи в CSV файл"""
        if not matches:
            return
            
        processed_matches = [self.process_match_data(match) for match in matches]
        df = pd.DataFrame(processed_matches)
        df.to_csv(filename, index=False)
        print(f"Сохранено {len(processed_matches)} матчей в {filename}")

# Пример использования
if __name__ == "__main__":
    collector = FootballDataCollector()
    
    # Получаем предстоящие матчи
    upcoming = collector.get_upcoming_matches(days_ahead=3)
    print(f"Найдено {len(upcoming)} предстоящих матчей")
    
    # Сохраняем в файл
    collector.save_matches_to_csv(upcoming, 'upcoming_matches.csv')
