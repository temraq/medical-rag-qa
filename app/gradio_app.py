import gradio as gr
import requests
import os
import logging

# 🎯 НАСТРОЙКА ЛОГИРОВАНИЯ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Gradio-UI")

API_URL = os.getenv("API_URL", "http://localhost:8000")
logger.info("API endpoint set to: %s", API_URL)

def query_rag(query):
    logger.info("📥 UI received query: %s", query)
    try:
        response = requests.post(f"{API_URL}/query", json={"query": query}, timeout=120)
        response.raise_for_status()
        result = response.json()
        answer = result.get("answer", "No answer returned.")
        logger.info("✅ Received answer from API (length: %d chars)", len(answer))
        return answer
    except requests.exceptions.Timeout:
        logger.error("⏰ Timeout connecting to API")
        return "Error: API request timed out (model may be loading)"
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Connection refused — is API running?")
        return "Error: Cannot connect to API. Check if server is running."
    except Exception as e:
        logger.error("💥 Unexpected error: %s", str(e))
        return f"Unexpected error: {str(e)}"

demo = gr.Interface(
    fn=query_rag,
    inputs=gr.Textbox(label="Enter your medical question"),
    outputs=gr.Textbox(label="Answer"),
    title="Medical QA RAG System",
    description="Ask a medical question and get answers based on PubMed literature."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)