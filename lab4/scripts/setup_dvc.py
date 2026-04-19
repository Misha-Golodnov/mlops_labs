#!/usr/bin/env python3
"""
Скрипт для автоматической настройки DVC remote.
Запустите один раз перед dvc pull.
"""

import subprocess
import sys

# ВАШИ ДАННЫЕ
CLIENT_ID = "766882206850-ra7idum35ugg2fg03istidv8sppinhe1.apps.googleusercontent.com"
CLIENT_SECRET = "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"


def run_command(cmd):

    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Ошибка: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True


def main():
    print("Настройка DVC remote для доступа к Google Drive...")
    print()

    # Используем python -m dvc вместо прямого вызова dvc
    if not run_command(f'python -m dvc remote modify myremote gdrive_client_id "{CLIENT_ID}"'):
        print("Ошибка при настройке client_id")
        sys.exit(1)

    if not run_command(f'python -m dvc remote modify myremote gdrive_client_secret "{CLIENT_SECRET}"'):
        print("Ошибка при настройке client_secret")
        sys.exit(1)

    print()
    print("DVC remote настроен!")
    print()
    print("Теперь выполните: python -m dvc pull")
    print("При первом запуске откроется браузер для авторизации в Google.")


if __name__ == "__main__":
    main()