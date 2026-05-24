FROM python:3.12-slim

# 1. Установка системных зависимостей для OpenCV и MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Установка рабочей директории
WORKDIR /app

# 3. Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Копируем исходный код
COPY src/ ./src/
COPY static/ ./static/
COPY main.py .

# 5. Создаем пустую папку для моделей
RUN mkdir -p models

# 6. Настройки порта и переменных окружения
EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# 7. Запуск сервера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]