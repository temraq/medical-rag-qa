# Dockerfile
FROM python:3.10-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Установка Google Cloud SDK для загрузки моделей
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание папок для моделей
RUN mkdir -p /app/models/zephyr_base_model \
    && mkdir -p /app/models/zephyr_medical_rag_adapter \
    && mkdir -p /app/index/pubmed_rag_index \
    && mkdir -p /app/offload \
    && mkdir -p /app/logs \
    && chmod -R 777 /app/models /app/index /app/offload /app/logs

# Загрузка моделей из Google Cloud Storage при сборке
ARG GCS_BUCKET_NAME
ARG GCS_MODEL_PATH
ARG GCS_ADAPTER_PATH
ARG GCS_INDEX_PATH

RUN if [ -n "$GCS_BUCKET_NAME" ]; then \
        echo "Загрузка моделей из GCS bucket: $GCS_BUCKET_NAME"; \
        gsutil -m cp -r "gs://$GCS_BUCKET_NAME/$GCS_MODEL_PATH/*" "/app/models/zephyr_base_model/" 2>/dev/null || echo "Модель не найдена, пропускаем"; \
        gsutil -m cp -r "gs://$GCS_BUCKET_NAME/$GCS_ADAPTER_PATH/*" "/app/models/zephyr_medical_rag_adapter/" 2>/dev/null || echo "Адаптер не найден, пропускаем"; \
        gsutil -m cp -r "gs://$GCS_BUCKET_NAME/$GCS_INDEX_PATH/*" "/app/index/pubmed_rag_index/" 2>/dev/null || echo "Индекс не найден, пропускаем"; \
    else \
        echo "GCS_BUCKET_NAME не указан, пропускаем загрузку моделей"; \
    fi

# Порт для Cloud Run
EXPOSE 8080

# Команда запуска
CMD ["python", "app.py"]