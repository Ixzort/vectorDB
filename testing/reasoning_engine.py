import os
from datetime import datetime
from openai import OpenAI
from search_system import search_posts
from prompts_config import PROMPT_TEMPLATES

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def detect_category(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ["возраст", "лет", "годов"]): return "age"
    if any(word in q for word in ["внешност", "выгляд", "видом"]): return "appearance"
    if any(word in q for word in ["эмоци", "чувств"]): return "emotions"
    if "повед" in q or "привыч" in q: return "behavior"
    if any(word in q for word in ["место", "локаци", "город", "страна"]): return "locations"
    if any(word in q for word in ["работ", "професс", "карьер", "учеб"]): return "work"
    if any(word in q for word in ["образ жизни", "хобби", "увлечен", "режим", "интерес"]): return "lifestyle"
    if any(word in q for word in ["друз", "семь", "окружен", "коллег"]): return "environment"
    if "настроен" in q: return "mood"
    return "personality"

def answer_query(question: str, index_name: str = "social-index", top_k: int = 5) -> str:
    category = detect_category(question)
    results = search_posts(question, index_name=index_name, top_k=top_k)
    print("DEBUG: найдено постов:", len(results))
    for r in results:
        print(">>", r)
    posts_context = ""
    for i, post in enumerate(results, start=1):
        date = post.get("date")
        text = post.get("text") or ""
        loc = post.get("location")
        if date and loc:
            posts_context += f"{i}) [{date}, {loc}] {text}\n"
        elif date:
            posts_context += f"{i}) [{date}] {text}\n"
        elif loc:
            posts_context += f"{i}) [{loc}] {text}\n"
        else:
            posts_context += f"{i}) {text}\n"
    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt_template = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["personality"])
    prompt_body = prompt_template.format(date=current_date, posts=posts_context.strip())
    full_prompt = f"Вопрос: {question}\n\n" + prompt_body
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": full_prompt}]
    )
    answer = response.choices[0].message.content
    return answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Анализ личности по социальным данным")
    parser.add_argument("-q", "--question", required=True, help="Вопрос о личности пользователя")
    parser.add_argument("-n", "--index", default="social-index", help="Имя Pinecone индекса с данными")
    parser.add_argument("-k", "--top", type=int, default=5, help="Максимальное число постов для контекста")
    args = parser.parse_args()
    reply = answer_query(args.question, index_name=args.index, top_k=args.top)
    print("Ответ модели:")
    print(reply)
