#!/usr/bin/env python3
"""
Главный файл для запуска футбольного бота прогнозов
"""

import os
import sys
import argparse
from multiprocessing import Process
import time

def run_web_app():
    """Запустить веб-приложение"""
    from app import app
    from config import Config
    
    print("🌐 Запуск веб-приложения...")
    app.run(
        host=Config.FLASK_HOST, 
        port=Config.FLASK_PORT, 
        debug=Config.DEBUG
    )

def run_telegram_bot():
    """Запустить Telegram бота"""
    from telegram_bot import main as telegram_main
    
    print("🤖 Запуск Telegram бота...")
    telegram_main()

def run_both():
    """Запустить и веб-приложение, и Telegram бота"""
    print("🚀 Запуск полной системы...")
    
    # Запускаем веб-приложение в отдельном процессе
    web_process = Process(target=run_web_app)
    web_process.start()
    
    # Небольшая задержка для запуска веб-приложения
    time.sleep(2)
    
    # Запускаем Telegram бота в основном процессе
    try:
        run_telegram_bot()
    except KeyboardInterrupt:
        print("\n⏹️ Остановка системы...")
        web_process.terminate()
        web_process.join()
        print("✅ Система остановлена")

def check_dependencies():
    """Проверить установленные зависимости"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'requests', 'beautifulsoup4', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Отсутствуют необходимые пакеты:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nУстановите их командой:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_config():
    """Проверить конфигурацию"""
    from config import Config
    
    issues = []
    
    if not Config.TELEGRAM_BOT_TOKEN:
        issues.append("TELEGRAM_BOT_TOKEN не установлен")
    
    if not Config.FOOTBALL_API_KEY or Config.FOOTBALL_API_KEY == 'your-api-key-here':
        issues.append("FOOTBALL_API_KEY не настроен")
    
    if issues:
        print("⚠️ Проблемы с конфигурацией:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nСоздайте файл .env на основе .env.example и настройте переменные")
        return False
    
    return True

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Футбольный Бот Прогнозов')
    parser.add_argument(
        '--mode', 
        choices=['web', 'telegram', 'both'], 
        default='both',
        help='Режим запуска: web (веб-приложение), telegram (бот), both (оба)'
    )
    parser.add_argument(
        '--check', 
        action='store_true',
        help='Проверить зависимости и конфигурацию'
    )
    parser.add_argument(
        '--no-check', 
        action='store_true',
        help='Запустить без проверок'
    )
    
    args = parser.parse_args()
    
    # Проверяем зависимости и конфигурацию
    if not args.no_check:
        print("🔍 Проверка системы...")
        
        if not check_dependencies():
            sys.exit(1)
        
        if not check_config():
            print("\n❓ Продолжить без полной конфигурации? (y/N): ", end='')
            if input().lower() != 'y':
                sys.exit(1)
        
        print("✅ Проверка завершена\n")
    
    if args.check:
        print("✅ Все проверки пройдены")
        return
    
    # Запускаем в выбранном режиме
    try:
        if args.mode == 'web':
            run_web_app()
        elif args.mode == 'telegram':
            run_telegram_bot()
        elif args.mode == 'both':
            run_both()
    except KeyboardInterrupt:
        print("\n⏹️ Остановка по запросу пользователя")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
