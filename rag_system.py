import logging
import os
from openai import OpenAI
from pinecone import Pinecone

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Конфигурация ключей и моделей
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "YOUR_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4"

# Новый способ инициализации Pinecone (v3.0.0+)
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("✅ Pinecone и OpenAI клиенты успешно инициализированы")
except Exception as e:
    logging.error(f"❌ Ошибка инициализации: {e}")
    pc = None
    index = None
    openai_client = None


def retrieve_relevant_posts(query: str, top_k: int = 5) -> str:
    """Выполняет векторный поиск по Pinecone и формирует текстовый контекст из найденных постов."""
    if not index or not openai_client:
        logging.error("Клиенты не инициализированы")
        return ""

    logging.info(f"Получен запрос: {query}")

    # 1. Генерация эмбеддинга запроса
    try:
        embedding_response = openai_client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_vector = embedding_response.data[0].embedding
    except Exception as e:
        logging.error(f"Ошибка при получении эмбеддинга запроса: {e}")
        return ""

    logging.info("Эмбеддинг запроса сгенерирован.")

    # 2. Поиск в Pinecone ближайших векторов
    try:
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        logging.error(f"Ошибка при запросе к Pinecone: {e}")
        return ""

    matches = results.get("matches", [])
    logging.info(f"Найдено результатов: {len(matches)}")

    # 3. Формирование контекста из результатов
    context_pieces = []
    used_post_ids = set()

    for match in matches:
        metadata = match.get("metadata", {})
        post_id = metadata.get("post_id") or match.get("id")

        if post_id in used_post_ids:
            continue
        used_post_ids.add(post_id)

        date = metadata.get("date")
        location = metadata.get("location")
        text = metadata.get("text")

        piece_parts = []
        if date:
            piece_parts.append(f"Дата: {date}.")
        if location:
            piece_parts.append(f"Локация: {location}.")
        if text:
            piece_parts.append(f"Пост: {text}")
        else:
            piece_parts.append("Пост: (текст недоступен)")

        piece = " ".join(piece_parts)
        context_pieces.append(piece)

    context_text = "\n\n".join(context_pieces)
    logging.info(f"Сформирован контекст для LLM (фрагментов: {len(context_pieces)})")
    return context_text


def generate_answer_with_llm(question: str, context_text: str) -> str:
    """Формирует prompt с контекстом и вопросом, запрашивает LLM и возвращает ответ."""
    if not openai_client:
        return "Ошибка: OpenAI клиент не инициализирован"

    if not context_text:
        logging.info("Контекст отсутствует, нет данных для ответа.")
        return "Извините, у меня недостаточно данных, чтобы ответить на этот вопрос."

    prompt = (
        "Используй следующую информацию для ответа на вопрос. "
        "Если ответа в информации нет, скажи, что не знаешь.\n\n"
        f"Информация:\n{context_text}\n\n"
        f"Вопрос: {question}\nОтвет:"
    )

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        logging.info("Ответ от LLM получен.")
        return answer
    except Exception as e:
        logging.error(f"Ошибка при генерации ответа LLM: {e}")
        return "Извините, произошла ошибка при попытке получить ответ от модели."


def answer_user_question(question: str) -> str:
    """Основная функция: выполняет поиск по постам и генерирует ответ на вопрос."""
    context = retrieve_relevant_posts(question)
    answer = generate_answer_with_llm(question, context)
    return answer


def test_connection():
    """Тестирует подключение к сервисам"""
    print("🔧 Тестирование подключений...")

    # Тест Pinecone
    if pc:
        try:
            stats = index.describe_index_stats()
            print(f"✅ Pinecone подключен. Векторов в индексе: {stats.get('total_vector_count', 0)}")
        except Exception as e:
            print(f"❌ Ошибка подключения к Pinecone: {e}")
    else:
        print("❌ Pinecone не инициализирован")

    # Тест OpenAI
    if openai_client:
        try:
            test_embedding = openai_client.embeddings.create(
                input=["test"],
                model=EMBEDDING_MODEL
            )
            print(f"✅ OpenAI подключен. Размерность эмбеддинга: {len(test_embedding.data[0].embedding)}")
        except Exception as e:
            print(f"❌ Ошибка подключения к OpenAI: {e}")
    else:
        print("❌ OpenAI не инициализирован")


if __name__ == "__main__":
    # Тестируем подключения
    test_connection()

    # Предупреждение о ключах
    if PINECONE_API_KEY == "YOUR_PINECONE_API_KEY" or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("\n⚠️  ВНИМАНИЕ: Замените YOUR_PINECONE_API_KEY и YOUR_OPENAI_API_KEY на реальные ключи!")
        print("Или установите переменные окружения:")
        print("export PINECONE_API_KEY='your_key_here'")
        print("export OPENAI_API_KEY='your_key_here'")
        print("export INDEX_NAME='your_index_name'")
    else:
        # Основной цикл
        print("\n🚀 Система готова к работе!")
        while True:
            try:
                user_question = input("\nВведите ваш вопрос (или 'quit' для выхода): ")
                if user_question.lower() in ['quit', 'exit', 'q']:
                    break
                answer = answer_user_question(user_question)
                print("\nОтвет:", answer)
            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
