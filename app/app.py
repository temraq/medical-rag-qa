from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from datetime import datetime
import logging
import os
import time
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

# 🎯 НАСТРОЙКА ЛОГИРОВАНИЯ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("/tmp/api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RAG-API")
app = FastAPI()

# Автоматическое определение окружения Cloud Run
IS_CLOUD_RUN = os.environ.get("K_SERVICE") is not None

# Конфигурация моделей (пути для загрузки из GCS)
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
MODEL_GCS_PATH = os.environ.get("MODEL_GCS_PATH", "zephyr-7b-base")
ADAPTERS_GCS_PATH = os.environ.get("ADAPTERS_GCS_PATH", "zephyr-medical-adapter")
INDEX_GCS_PATH = os.environ.get("INDEX_GCS_PATH", "pubmed-rag-index")

# Проверка обязательных переменных окружения
REQUIRED_ENV_VARS = ["GCS_BUCKET_NAME"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
if missing_vars and IS_CLOUD_RUN:
    error_msg = f"❌ Отсутствуют обязательные переменные окружения: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise EnvironmentError(error_msg)

# Локальные пути внутри контейнера
BASE_DIR = "/app"
MODEL_LOCAL_PATH = f"{BASE_DIR}/models/zephyr_base_model"
ADAPTERS_LOCAL_PATH = f"{BASE_DIR}/models/zephyr_medical_rag_adapter"
INDEX_LOCAL_PATH = f"{BASE_DIR}/index/pubmed_rag_index"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MIN_RELEVANCE_THRESHOLD = 0.55
MAX_CONTEXT_LENGTH = 512 if not IS_CLOUD_RUN else 384  # Уменьшаем для Cloud Run

logger.info(f"🚀 Запуск RAG API сервера в режиме: {'Cloud Run' if IS_CLOUD_RUN else 'Local'}")
logger.info(f"📦 Bucket: {GCS_BUCKET_NAME}, Model path: {MODEL_GCS_PATH}")

def download_from_gcs(bucket_name, gcs_path, local_path):
    """Загружает файлы из Google Cloud Storage с использованием Python SDK"""
    if not bucket_name:
        logger.warning("GCS_BUCKET_NAME не указан, пропускаем загрузку")
        return False
    
    try:
        os.makedirs(local_path, exist_ok=True)
        logger.info(f"⬇️ Загрузка из gs://{bucket_name}/{gcs_path} в {local_path}")
        
        # Инициализация клиента GCS
        try:
            storage_client = storage.Client()
            logger.info("✅ GCS клиент инициализирован успешно")
        except DefaultCredentialsError as e:
            logger.error(f"❌ Ошибка аутентификации GCS: {e}")
            logger.info("🔄 Попытка использовать анонимный доступ...")
            storage_client = storage.Client.create_anonymous_client()
        
        bucket = storage_client.bucket(bucket_name)
        
        # Получение списка всех файлов в директории
        blobs = bucket.list_blobs(prefix=f"{gcs_path}/")
        total_files = 0
        downloaded_files = 0
        
        start_time = time.time()
        
        for blob in blobs:
            total_files += 1
            # Пропускаем саму директорию
            if blob.name.endswith('/'):
                continue
                
            # Создаем локальный путь
            relative_path = blob.name[len(gcs_path):].lstrip('/')
            local_file_path = os.path.join(local_path, relative_path)
            
            # Создаем директории если нужно
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Скачиваем файл
            logger.info(f"📥 Скачивание: {blob.name} -> {local_file_path}")
            blob.download_to_filename(local_file_path)
            downloaded_files += 1
            logger.info(f"✅ Загружено: {local_file_path}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Загрузка завершена: {downloaded_files}/{total_files} файлов за {elapsed_time:.2f} секунд")
        return downloaded_files > 0
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки из GCS: {e}", exc_info=True)
        return False

def ensure_models_available():
    """Гарантирует наличие всех моделей и индекса"""
    logger.info("🔍 Проверка наличия моделей и индекса...")
    
    # Загружаем базовую модель
    if not os.path.exists(MODEL_LOCAL_PATH) or not os.listdir(MODEL_LOCAL_PATH):
        logger.info("🔄 Базовая модель отсутствует - загружаем из GCS")
        success = download_from_gcs(GCS_BUCKET_NAME, MODEL_GCS_PATH, MODEL_LOCAL_PATH)
        if not success:
            logger.error("❌ Не удалось загрузить базовую модель")
    
    # Загружаем адаптеры
    if not os.path.exists(ADAPTERS_LOCAL_PATH) or not os.listdir(ADAPTERS_LOCAL_PATH):
        logger.info("🔄 Адаптеры отсутствуют - загружаем из GCS")
        success = download_from_gcs(GCS_BUCKET_NAME, ADAPTERS_GCS_PATH, ADAPTERS_LOCAL_PATH)
        if not success:
            logger.error("❌ Не удалось загрузить адаптеры")
    
    # Загружаем индекс
    if not os.path.exists(INDEX_LOCAL_PATH) or not os.listdir(INDEX_LOCAL_PATH):
        logger.info("🔄 FAISS индекс отсутствует - загружаем из GCS")
        success = download_from_gcs(GCS_BUCKET_NAME, INDEX_GCS_PATH, INDEX_LOCAL_PATH)
        if not success:
            logger.error("❌ Не удалось загрузить FAISS индекс")
    
    # Проверяем результат
    paths = {
        "базовая модель": MODEL_LOCAL_PATH,
        "адаптеры": ADAPTERS_LOCAL_PATH,
        "индекс": INDEX_LOCAL_PATH
    }
    
    missing = [name for name, path in paths.items() if not os.path.exists(path) or not os.listdir(path)]
    
    if missing:
        error_msg = f"❌ Не удалось загрузить компоненты: {', '.join(missing)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✅ Все компоненты успешно загружены")

# Загрузка моделей перед запуском API
logger.info("⏳ Начинаем загрузку моделей...")
start_time = time.time()
ensure_models_available()
logger.info(f"✅ Модели загружены за {time.time() - start_time:.2f} секунд")

# Инициализация моделей
tokenizer = None
model = None
vector_db = None

logger.info("\n📦 Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
logger.info("✅ Токенизатор загружен.")

logger.info("\n🤖 Загрузка модели в 4-bit для Cloud Run...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Оптимизация для Cloud Run - используем CPU offload и меньше памяти
model = AutoModelForCausalLM.from_pretrained(
    MODEL_LOCAL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="/tmp/offload",
    trust_remote_code=False,
    use_cache=True,
    low_cpu_mem_usage=True
)

logger.info("🔧 Применение адаптеров...")
model = PeftModel.from_pretrained(model, ADAPTERS_LOCAL_PATH)
model.eval()
logger.info("✅ Модель и адаптеры загружены.")

logger.info("🌐 Загрузка модели эмбеддингов...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

logger.info(f"Loading FAISS index from {INDEX_LOCAL_PATH}...")
vector_db = FAISS.load_local(
    INDEX_LOCAL_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)
logger.info("✅ FAISS index loaded successfully.")

# Функция smart_retrieve (без изменений)
def smart_retrieve(query, k=3):
    base_results = vector_db.similarity_search_with_relevance_scores(query, k=20)
    doc_to_score = {doc.metadata["id"]: score for doc, score in base_results}

    query_lower = query.lower()
    medical_guideline_keywords = ["guideline", "standard", "recommendation", "protocol", "consensus", "algorithm"]
    diagnostic_keywords = ["diagnos", "criteria", "symptom", "sign", "test", "screening"]
    treatment_keywords = ["treat", "therapy", "management", "intervention", "medication", "drug"]

    is_guideline = any(kw in query_lower for kw in medical_guideline_keywords)
    is_diagnostic = any(kw in query_lower for kw in diagnostic_keywords)
    is_treatment = any(kw in query_lower for kw in treatment_keywords)
    token_count = len(tokenizer.encode(query))

    if is_guideline:
        lam = 0.98
        reason = "Клиническая рекомендация"
    elif is_diagnostic or is_treatment:
        lam = 0.88
        reason = "Диагностика/лечение"
    elif token_count > 40:
        lam = 0.75
        reason = "Длинный запрос (>40 токенов)"
    else:
        lam = 0.82
        reason = "Общий запрос"

    logger.info(f"Определен тип запроса: {reason}, lambda = {lam:.2f}")

    mmr_results = vector_db.max_marginal_relevance_search(
        query,
        k=k,
        lambda_mult=lam,
        fetch_k=20
    )

    unique_results = {}
    for doc in mmr_results:
        pub_id = doc.metadata.get("pubmed_id", doc.metadata.get("id", "unknown"))
        if pub_id in unique_results:
            continue
        score = doc_to_score.get(doc.metadata["id"], 0.0)
        unique_results[pub_id] = (doc, score)
        if len(unique_results) >= k:
            break

    sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
    docs = [item[0] for item in sorted_results]
    scores = [item[1] for item in sorted_results]

    return docs, scores

# Функция форматирования сообщений (без изменений)
def format_zephyr_rag_messages(query, retrieved_docs):
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        metadata = doc.metadata or {}
        source_info = f"[Source {i+1}]"
        if "title" in metadata:
            source_info += f" '{metadata['title']}'"
        if "pubmed_id" in metadata:
            source_info += f" (PMID: {metadata['pubmed_id']})"
        content = doc.page_content if hasattr(doc, "page_content") else getattr(doc, "content", str(doc))
        context_parts.append({"source": source_info, "content": content, "metadata": metadata})

    assembled_context = "\n\n".join(f"{p['source']}\n{p['content']}" for p in context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical expert analyzing evidence. Your task is to:"
                " 1. Identify specific findings in the context relevant to the question"
                " 2. Note statistical significance (p-values, confidence intervals) when present"
                " 3. Reference source numbers when making claims"
                " 4. If answer isn't explicit, state 'I cannot provide a definitive answer'"
                " Do NOT use prior knowledge - base your answer ONLY on the provided context."
                f" Current date: {datetime.now().strftime('%Y-%m-%d')}"
            )
        },
        {
            "role": "user",
            "content": f"Medical Context:\n{assembled_context}\n\nQuestion: {query}"
        }
    ]
    return messages, assembled_context, context_parts

# Класс для остановки генерации (без изменений)
class StopOnSubsequences(StoppingCriteria):
    def __init__(self, stop_sequences_ids):
        self.stop_sequences_ids = [seq for seq in stop_sequences_ids if seq]

    def _ends_with(self, haystack, needle):
        n = len(needle)
        if n == 0 or len(haystack) < n:
            return False
        return haystack[-n:] == needle

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last = input_ids[0].tolist()
        for seq in self.stop_sequences_ids:
            if self._ends_with(last, seq):
                return True
        return False

# Основная функция RAG-пайплайна (оптимизированная для Cloud Run)
def zephyr_rag_pipeline(query: str, k: int = 3, max_new_tokens: int = 350, min_relevance: float = MIN_RELEVANCE_THRESHOLD):
    if IS_CLOUD_RUN:
        k = min(k, 2)  # Ограничиваем количество документов в Cloud Run
        max_new_tokens = min(max_new_tokens, 150)  # Уменьшаем длину генерации

    retrieved_docs, scores = smart_retrieve(query, k=k)
    if not scores or max(scores) < min_relevance:
        return {
            "answer": "Based on the available medical sources, I cannot give a definitive answer to this question.",
            "confidence": "low",
            "retrieved_docs": [],
            "scores": scores,
            "assembled_context": "",
            "context_parts": []
        }

    messages, assembled_context, context_parts = format_zephyr_rag_messages(query, retrieved_docs)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH,
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    eos_token_id = tokenizer.eos_token_id

    stop_markers = [
        "<|end|>", "<|user|>", "<|assistant|>", "</s>",
        "<| user|>", "<|Assistant|>", "<||user|>", "<|User|>"
    ]
    stop_ids_list = []
    for sm in stop_markers:
        try:
            ids = tokenizer.encode(sm, add_special_tokens=False)
            if ids:
                stop_ids_list.append(ids)
        except Exception:
            continue

    stopping_criteria = StoppingCriteriaList([StopOnSubsequences(stop_ids_list)])
    max_new_tokens = min(max_new_tokens, 256)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.8,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    full_ids = outputs[0].tolist()
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = full_ids[prompt_len:]

    assistant_ids = tokenizer.encode("<|assistant|>", add_special_tokens=False)
    def find_subsequence_last(haystack, needle):
        if not needle:
            return -1
        n = len(needle)
        for i in range(len(haystack) - n, -1, -1):
            if haystack[i:i + n] == needle:
                return i
        return -1

    start_idx = 0
    if assistant_ids:
        pos = find_subsequence_last(gen_ids, assistant_ids)
        if pos != -1:
            start_idx = pos + len(assistant_ids)

    end_idx = None
    for stop_ids in stop_ids_list:
        n = len(stop_ids)
        for i in range(start_idx, len(gen_ids) - n + 1):
            if gen_ids[i:i + n] == stop_ids:
                if end_idx is None or i < end_idx:
                    end_idx = i
                break

    if end_idx is None:
        answer_ids = gen_ids[start_idx:]
    else:
        answer_ids = gen_ids[start_idx:end_idx]

    answer = tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

    end_marker_phrase = "This conclusion is based strictly on the specific medical evidence provided in the context."
    low_ans = answer.lower()
    if end_marker_phrase.lower() in low_ans:
        idx = low_ans.index(end_marker_phrase.lower()) + len(end_marker_phrase)
        answer = answer[:idx].strip()

    first_pos = None
    for sm in stop_markers:
        pos = answer.find(sm)
        if pos != -1:
            if first_pos is None or pos < first_pos:
                first_pos = pos
    if first_pos is not None:
        answer = answer[:first_pos].strip()

    if not answer:
        answer = "Based on the available medical sources, I cannot give a definitive answer to this question."

    context_used = False
    try:
        for i in range(1, k + 1):
            if f"source {i}" in answer.lower() or f"[source {i}]" in answer.lower():
                context_used = True
                break
    except Exception:
        context_used = False

    confidence = "high" if (max(scores) > 0.75 and context_used) else "medium"
    full_generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    return {
        "answer": answer,
        "confidence": confidence,
        "retrieved_docs": retrieved_docs,
        "scores": scores,
        "full_generated_text": full_generated_text,
        "context_used": context_used,
        "assembled_context": assembled_context,
        "context_parts": context_parts
    }

# Эндпоинт для API
class QueryRequest(BaseModel):
    query: str
    k: int = 3
    min_relevance: float = MIN_RELEVANCE_THRESHOLD

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        logger.info(f"🔍 Получен запрос: {request.query}")
        start_time = time.time()
        
        result = zephyr_rag_pipeline(
            query=request.query,
            k=request.k,
            min_relevance=request.min_relevance
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Запрос обработан за {processing_time:.2f} секунд, confidence: {result['confidence']}")
        
        return result
    except Exception as e:
        logger.exception(f"❌ Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "healthy",
        "models_loaded": all([
            tokenizer is not None,
            model is not None,
            vector_db is not None
        ]),
        "environment": "Cloud Run" if IS_CLOUD_RUN else "Local",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8081))
    logger.info(f"🚀 Запуск сервера на порту {port}")
    
    # Добавляем startup event для логгирования
    @app.on_event("startup")
    async def startup_event():
        logger.info("✅ API сервер запущен и готов принимать запросы")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        timeout_keep_alive=300  # Увеличиваем таймаут для Cloud Run
    )