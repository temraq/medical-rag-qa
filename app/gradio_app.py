import gradio as gr
import requests
import os
import logging

# 🎯 НАСТРОЙКА ЛОГИРОВАНИЯ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("/tmp/ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Gradio-UI")

# Автоматическое определение URL API в Cloud Run
API_URL = os.environ.get("API_URL", "http://localhost:8000")
if "GCP_PROJECT" in os.environ:  # Cloud Run environment
    API_URL = f"http://{os.environ.get('K_SERVICE')}:8000"

logger.info("🌐 UI подключен к API: %s", API_URL)

def query_rag(query):
    logger.info("📥 Получен запрос: %s", query)
    try:
        response = requests.post(f"{API_URL}/query", json={"query": query}, timeout=180)
        response.raise_for_status()
        result = response.json()
        answer = result.get("answer", "❌ Не удалось получить ответ от модели")
        logger.info("✅ Получен ответ (длина: %d символов)", len(answer))
        return answer
    except requests.exceptions.Timeout:
        logger.error("⏰ Таймаут запроса к API")
        return "❌ Таймаут: модель может загружаться или перегружена"
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Нет подключения к API")
        return "❌ Нет подключения к API. Проверьте статус сервиса."
    except Exception as e:
        logger.error("💥 Неожиданная ошибка: %s", str(e))
        return f"❌ Ошибка: {str(e)}"

demo = gr.Interface(
    fn=query_rag,
    inputs=gr.Textbox(label="📝 Ваш медицинский вопрос", placeholder="Например: Какие первые признаки диабета 2 типа?"),
    outputs=gr.Textbox(label="🧠 Ответ системы"),
    title="🏥 Medical QA RAG System",
    description="Система отвечает на медицинские вопросы на основе научных статей PubMed",
    examples=[
        ["Какие симптомы диабета 2 типа?"],
        ["Какое первое лекарство назначают при гипертонии?"],
        ["Каковы рекомендации по скринингу рака молочной железы?"]
    ],
    css="""
    .gradio-container { max-width: 800px; margin: 0 auto; }
    h1 { color: #0d6efd; text-align: center; }
    .footer { text-align: center; margin-top: 20px; color: #666; }
    """
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Отключаем public sharing в продакшене
        auth=None    # Отключаем аутентификацию для демо
    )