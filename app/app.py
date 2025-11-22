from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from datetime import datetime
import logging
import time
from tqdm import tqdm
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig


# 🎯 НАСТРОЙКА ЛОГИРОВАНИЯ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RAG-API")
app = FastAPI()

# Конфигурация модели
MODEL_NAME = "models/zephyr_base_model"
ADAPTERS_PATH = "models/zephyr_medical_rag_adapter"
INDEX_PATH = "index/pubmed_rag_index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MIN_RELEVANCE_THRESHOLD = 0.55
MAX_CONTEXT_LENGTH = 512

logger.info("Starting RAG API server...")

def create_progress_bar(total_steps, desc="Loading"):
    """Создает прогресс-бар для отслеживания загрузки"""
    return tqdm(total=total_steps, desc=desc, unit="step", colour='green')

try:
    logger.info("\n📦 Загрузка токенизатора...")
    pbar_tokenizer = create_progress_bar(2, desc="Loading tokenizer")

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pbar_tokenizer.update(1)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pbar_tokenizer.update(1)
    pbar_tokenizer.close()
    
    logger.info("✅ Токенизатор загружен.")
    logger.info("\n🤖 Загрузка модели на CPU с offloading на диск...")
    pbar_model = create_progress_bar(3, desc="Loading model")

    config = AutoConfig.from_pretrained(MODEL_NAME)

    # Определяем, сколько RAM доступно (например, 6 ГБ для модели, остальное — системе)
    # Если у вас мало RAM (<16 ГБ), уменьшите "cpu" до "4GiB"
    max_memory = {
        "cpu": "4GiB",      # ← подстройте под вашу систему
        "disk": "20GiB"     # виртуальная "память" на диске
    }

    # Автоматическое распределение слоёв между CPU и диском
    device_map = infer_auto_device_map(
        AutoModelForCausalLM.from_config(config),
        max_memory=max_memory,
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype=torch.float16  # или torch.bfloat16, если поддерживается
    )

    logger.info(f"Сгенерирован device_map: {device_map}")

    # Загружаем модель БЕЗ квантизации
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        offload_folder="offload",
        offload_state_dict=True,
        torch_dtype=torch.float16,  # экономия памяти vs float32
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )
    pbar_model.update(2)

    # Применяем адаптеры (LoRA) — они работают на CPU
    logger.info("🔧 Применение адаптеров...")
    model = PeftModel.from_pretrained(model, ADAPTERS_PATH)
    model.eval()
    pbar_model.update(1)
    pbar_model.close()
    logger.info("✅ Модель и адаптеры загружены на CPU с offloading.")
    
    # Загрузка FAISS индекса (без изменений)
    logger.info("🌐 Загрузка модели эмбеддингов...")
    pbar_embeddings = create_progress_bar(2, desc="Loading embeddings")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    pbar_embeddings.update(1)
    logger.info("Loading FAISS index from %s...", INDEX_PATH)
    pbar_faiss = create_progress_bar(1, desc="Loading FAISS index")
    vector_db = FAISS.load_local(
        INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    pbar_faiss.update(1)
    pbar_faiss.close()
    pbar_embeddings.close()
    
except Exception as e:
    logger.exception(f"Initialization failed: {e}")  # ← лучше логировать traceback

# Функция smart_retrieve (адаптированная для CPU)
def smart_retrieve(query, k=3):
    """
    Параметры:
    - query: текстовый запрос
    - k: количество результатов для возврата

    Возвращает:
    - Список документов
    - Список соответствующих скоров релевантности
    """
    # Получаем базовые результаты с их расстояниями

    base_results = vector_db.similarity_search_with_relevance_scores(query, k=20)
    doc_to_score = {doc.metadata["id"]: score for doc, score in base_results}

    # Определяем тип запроса и устанавливаем оптимальный параметр lambda
    query_lower = query.lower()

  # 1. Более точное определение типов запросов
    medical_guideline_keywords = ["guideline", "standard", "recommendation", "protocol", "consensus", "algorithm"]
    diagnostic_keywords = ["diagnos", "criteria", "symptom", "sign", "test", "screening"]
    treatment_keywords = ["treat", "therapy", "management", "intervention", "medication", "drug"]

    is_guideline = any(kw in query_lower for kw in medical_guideline_keywords)
    is_diagnostic = any(kw in query_lower for kw in diagnostic_keywords)
    is_treatment = any(kw in query_lower for kw in treatment_keywords)

  # 2. Используем токены вместо слов для оценки длины
    token_count = len(tokenizer.encode(query))

    # 3. Оптимизированные значения lambda для медицинских сценариев
    if is_guideline:
        # Для клинических рекомендаций нужна МАКСИМАЛЬНАЯ точность
        # 0.98 оставляет всего 2% на разнообразие (минимум необходимого)
        lam = 0.98
        reason = "Клиническая рекомендация"

    elif is_diagnostic or is_treatment:
        # Для диагностических и лечебных запросов важна точность, но нужен некоторый контекст
        lam = 0.88
        reason = "Диагностика/лечение"

    elif token_count > 40:  # Более разумный порог (40 токенов вместо 8 слов)
        # Для действительно длинных запросов
        lam = 0.75
        reason = "Длинный запрос (>40 токенов)"

    else:
        # Для коротких и общих запросов
        lam = 0.82
        reason = "Общий запрос"

    print(f"Определен тип запроса: {reason}, lambda = {lam:.2f}")

    # Улучшенный MMR с большим fetch_k для лучшего разнообразия
    mmr_results = vector_db.max_marginal_relevance_search(
        query,
        k=k,
        lambda_mult=lam,
        fetch_k=20  # Увеличиваем для лучшего выбора
    )

    # Убираем дубли из одной статьи (макс 1 чанк на статью)
    unique_results = {}
    for doc in mmr_results:
        # ИСПРАВЛЕНИЕ: Используем pubmed_id как основной идентификатор статьи
        pub_id = doc.metadata.get("pubmed_id", doc.metadata.get("id", "unknown"))

        # Если статья уже добавлена, пропускаем
        if pub_id in unique_results:
            continue

        # Получаем скор для этого документа
        score = doc_to_score.get(doc.metadata["id"], 0.0)
        unique_results[pub_id] = (doc, score)

        # Прекращаем, когда набрали достаточно уникальных статей
        if len(unique_results) >= k:
            break

    # Сортируем результаты по скору перед возвратом
    sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)

    # Разделяем результаты и скоры
    docs = [item[0] for item in sorted_results]
    scores = [item[1] for item in sorted_results]

    return docs, scores

# Функция форматирования сообщений
def format_zephyr_rag_messages(query, retrieved_docs):
    """Возвращает (messages, assembled_context_str, context_parts_list)"""
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        metadata = doc.metadata or {}
        source_info = f"[Source {i+1}]"
        if "title" in metadata:
            source_info += f" '{metadata['title']}'"
        if "pubmed_id" in metadata:
            source_info += f" (PMID: {metadata['pubmed_id']})"

        content = doc.page_content if hasattr(doc, "page_content") else getattr(doc, "content", str(doc))
        # не усекать здесь — собираем полный кусок; тримить можно при печати
        context_parts.append({"source": source_info, "content": content, "metadata": metadata})

    # собираем текст контекста в одну строку (для логирования / токенизации)
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

# Класс для остановки генерации
class StopOnSubsequences(StoppingCriteria):
    """
    Останавливает генерацию, когда последний сгенерированный фрагмент
    оканчивается на любую из переданных подпоследовательностей токенов.
    """
    def __init__(self, stop_sequences_ids):
        # stop_sequences_ids: list of list of ints
        self.stop_sequences_ids = [seq for seq in stop_sequences_ids if seq]  # фильтруем пустые

    def _ends_with(self, haystack, needle):
        n = len(needle)
        if n == 0 or len(haystack) < n:
            return False
        return haystack[-n:] == needle

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Работает с первым (и обычно единственным) элементом батча
        last = input_ids[0].tolist()
        for seq in self.stop_sequences_ids:
            if self._ends_with(last, seq):
                return True
        return False

# Основная функция RAG-пайплайна
def zephyr_rag_pipeline(query: str, k: int = 3, max_new_tokens: int = 350, min_relevance: float = MIN_RELEVANCE_THRESHOLD):
    """
    Возвращает: answer, confidence, retrieved_docs, scores, full_generated_text, context_used,
                assembled_context (строка), context_parts (list of dicts with source/content/metadata)
    """
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

    # Получаем теперь тройку: messages, assembled_context и отдельные части
    messages, assembled_context, context_parts = format_zephyr_rag_messages(query, retrieved_docs)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Токенизация промпта
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # гарантируем pad token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    eos_token_id = tokenizer.eos_token_id

    # сбор стоп-маркеров как у вас был
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

    # find nearest stop
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

    # постобработка как у вас была
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

    # определяем использование источников
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
        result = zephyr_rag_pipeline(
            query=request.query,
            k=request.k,
            min_relevance=request.min_relevance
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)