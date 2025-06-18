import os
import json
import time
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Загрузка переменных окружения
load_dotenv()

# Инициализация Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Параметры индекса
INDEX_NAME = "posts-text-index"
DIMENSION = 1536

# Подключение к существующему индексу
index = pc.Index(INDEX_NAME)

# Инициализация модели эмбеддингов
openai_api_key = os.getenv("OPENAI_API_KEY")
embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=DIMENSION,
    openai_api_key=openai_api_key
)


# Улучшенная функция очистки метаданных
def deep_clean_metadata(metadata):
    """
    Рекурсивно очищает метаданные:
    - Заменяет None на пустую строку
    - Преобразует списки в строки
    - Удаляет вложенные структуры
    - Проверяет типы данных
    """
    cleaned = {}
    for key, value in metadata.items():
        # Обработка None значений
        if value is None:
            cleaned[key] = ""
            continue

        # Обработка списков
        if isinstance(value, list):
            # Фильтрация None в списках
            filtered_list = [item for item in value if item is not None]
            cleaned[key] = ",".join(str(item) for item in filtered_list)
            continue

        # Обработка словарей (недопустимы в Pinecone)
        if isinstance(value, dict):
            # Преобразуем вложенный словарь в строку JSON
            cleaned[key] = json.dumps(value)
            continue

        # Проверка допустимых типов
        if not isinstance(value, (str, int, float, bool)):
            try:
                # Попытка преобразования в строку
                cleaned[key] = str(value)
            except:
                cleaned[key] = ""
            continue

        # Все остальные случаи
        cleaned[key] = value

    return cleaned


# Загрузка обработанных постов
def load_processed_posts():
    try:
        with open('output_vector.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {str(e)}")
        return []


# Функция для проверки метаданных перед загрузкой
def validate_metadata(metadata):
    """Проверяет метаданные на соответствие требованиям Pinecone"""
    for key, value in metadata.items():
        # Проверка типа значения
        if not isinstance(value, (str, int, float, bool)) and not (
                isinstance(value, list) and all(isinstance(item, str) for item in value)):
            print(f"⚠️ Недопустимый тип для поля '{key}': {type(value)}")
            return False

        # Проверка на None
        if value is None:
            print(f"⚠️ Обнаружено None значение для поля '{key}'")
            return False

    return True


# Основная функция
def main():
    # Загрузка данных
    processed_posts = load_processed_posts()
    if not processed_posts:
        print("Нет данных для обработки")
        return

    print(f"Загружено постов: {len(processed_posts)}")

    # Подготовка данных для загрузки
    vectors_to_upsert = []
    error_count = 0

    for post in processed_posts:
        post_id = post["id"]
        text = post["text"]

        # Генерация эмбеддинга для текста
        try:
            embedding = embed_model.embed_query(text)
            if len(embedding) != DIMENSION:
                print(f"⚠️ Размерность эмбеддинга ({len(embedding)}) не соответствует требуемой ({DIMENSION})")
                continue
        except Exception as e:
            print(f"❌ Ошибка генерации эмбеддинга для поста {post_id}: {str(e)}")
            continue

        # Базовые метаданные
        metadata = {
            "post_id": post_id,
            "text": text[:500] + "..." if len(text) > 500 else text
        }

        # Добавление дополнительных метаданных с глубокой очисткой
        if "metadata" in post:
            metadata.update(deep_clean_metadata(post["metadata"]))

        # Дополнительная очистка конкретных проблемных полей
        for field in ["activity_type", "sentiment", "location"]:
            if field in metadata and metadata[field] is None:
                metadata[field] = ""

        # Валидация метаданных перед добавлением
        if not validate_metadata(metadata):
            print(f"❌ Невалидные метаданные для поста {post_id}. Пропускаем.")
            error_count += 1
            # Сохраняем проблемный пост для анализа
            with open(f"error_post_{post_id}.json", "w") as f:
                json.dump({"post": post, "metadata": metadata}, f, indent=2)
            continue

        vectors_to_upsert.append({
            "id": post_id,
            "values": embedding,
            "metadata": metadata
        })

    print(f"Подготовлено векторов: {len(vectors_to_upsert)}")
    print(f"Обнаружено ошибок: {error_count}")

    # Загрузка данных по одному для точной диагностики
    success_count = 0
    for i, vector_data in enumerate(vectors_to_upsert):
        try:
            # Подробная информация о векторе перед загрузкой
            print(f"\nЗагрузка поста {i + 1}/{len(vectors_to_upsert)}: {vector_data['id']}")
            print("Метаданные:", {k: type(v) for k, v in vector_data["metadata"].items()})

            # Загрузка одного вектора
            index.upsert(vectors=[vector_data])
            success_count += 1
            print(f"✅ Успешно загружен")

        except Exception as e:
            print(f"❌ Ошибка при загрузке: {str(e)}")
            # Сохраняем проблемный вектор
            with open(f"error_vector_{vector_data['id']}.json", "w") as f:
                json.dump(vector_data, f, indent=2)
            print("Проблемный вектор сохранен в файл")

        # Пауза между запросами
        time.sleep(0.1)

    print(f"\nИтоги загрузки:")
    print(f"• Успешно загружено: {success_count}")
    print(f"• Ошибок: {len(vectors_to_upsert) - success_count}")

    # Проверка загрузки
    try:
        stats = index.describe_index_stats()
        print("\n📊 Статистика индекса:")
        print(f"• Всего векторов: {stats['total_vector_count']}")
        print(f"• Размерность: {stats['dimension']}")

        # Тестовый запрос
        if stats['total_vector_count'] > 0:
            print("\n🧪 Выполнение тестового запроса...")
            test_vector = vectors_to_upsert[0]["values"]
            test_results = index.query(
                vector=test_vector,
                top_k=3,
                include_metadata=True
            )

            print("🔍 Результаты тестового запроса:")
            for match in test_results["matches"]:
                print(f"• ID: {match['id']}, Сходство: {match['score']:.4f}")
                print(f"  Метаданные: {list(match['metadata'].keys())}")
    except Exception as e:
        print(f"❌ Ошибка при получении статистики: {str(e)}")


if __name__ == "__main__":
    main()