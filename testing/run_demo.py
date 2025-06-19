import argparse
from reasoning_engine import answer_query

parser = argparse.ArgumentParser(description="Демонстрация системы анализа личности (Chain-of-Thought)")
parser.add_argument("-q", "--question", required=True, help="Вопрос о личности пользователя")
parser.add_argument("-n", "--index", default="social_index", help="Имя Pinecone индекса с данными постов")
parser.add_argument("-k", "--top", type=int, default=5, help="Максимальное число постов для контекста")
args = parser.parse_args()

reply = answer_query(args.question, index_name=args.index, top_k=args.top)
print("Ответ модели:")
print(reply)

# Сохраняем ответ в файл (добавляет каждый новый ответ в конец)
with open("output_gpt.txt", "a", encoding="utf-8") as f:
    f.write(f"Вопрос: {args.question}\n")
    f.write(f"Ответ: {reply}\n")
    f.write("="*50 + "\n")