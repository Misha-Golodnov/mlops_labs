FROM python:3.10-slim

WORKDIR /app

#устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем папку с кодом FastApi
COPY ./app ./app

# Копируем модель
COPY ./data/processed ./data/processed

EXPOSE 8000

# Запускаем FastApi

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

