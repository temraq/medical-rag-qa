# Dockerfile
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgl1 \
    # Установка Google Cloud SDK
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/*


# Рабочая директория
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p /app/models/zephyr_base_model \
    && mkdir -p /app/models/zephyr_medical_rag_adapter \
    && mkdir -p /app/index/pubmed_rag_index \
    && mkdir -p /tmp/offload \
    && chmod -R 777 /app /tmp

# Порт для Cloud Run
EXPOSE 8080

# Запуск приложения
CMD ["python", "app.py"]