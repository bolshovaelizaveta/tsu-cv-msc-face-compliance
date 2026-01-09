FROM python:3.12-slim

# Системные зависимости для OpenCV и MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Установка зависимостей 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Исходный код 
COPY src/ ./src/
COPY models/ ./models/
COPY static/ ./static/
COPY main.py .

# Порт для FastAPI
EXPOSE 8000

# Команда для запуска 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]