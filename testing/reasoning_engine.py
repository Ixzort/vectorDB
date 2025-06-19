import os, json, logging
from datetime import datetime, timedelta
from openai import OpenAI
from search_system import search_posts
from prompts_config import PROMPT_TEMPLATES

logging.basicConfig(filename="app.log", level=logging.INFO)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

CACHE_PATH = "cache.json"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

def detect_category(question: str) -> str:
    # (оставить как в твоей версии)
    q = question.lower()
    if any(word in q for word in ["возраст", "лет", "годов"]): return "age"
    if any(word in q for word in ["внешност", "выгляд", "видом"]): return "appearance"
    if any(word in q for word in ["эмоци", "чувств"]): return "mood"
    if "повед" in q: return "personality"
    if "привыч" in q or "хобби" in q or "увлечен" in q or "интерес" in q or "режим" in q: return "lifestyle"
    if any(word in q for word in ["работ", "професс", "карьер", "учеб"]): return "work"
    if any(word in q for word in ["место", "локаци", "город", "страна", "живет", "жилье"]): return "locations"
    if "настроен" in q: return "mood"
    if any(word in q for word in ["друз", "семь", "окружен", "коллег"]): return "environment"
    if any(word in q for word in ["подруж", "подружиться", "познаком"]): return "friendship"
    if any(word in q for word in ["отдыхает", "отдых", "отпуск", "отдых проводит", "отпуске"]): return "locations"
    if "где" in q and "найти" in q: return "locations"
    if any(word in q for word in ["девушк", "парень", "парня", "girlfriend", "бойфренд", "отношен", "встречает", "жена", "муж"]): return "relationship"
    if any(word in q for word in ["дети", "ребенок", "ребёнок", "сын", "дочь"]): return "children"
    return "personality"

def answer_query(question: str, index_name: str = "social-index", top_k: int = 5) -> str:
    key = question
    if key in cache:
        logging.info(f"question={question} [CACHED]")
        return cache[key]

    category = detect_category(question)
    results = search_posts(question, index_name=index_name, top_k=top_k)
    if not results or len(results) == 0:
        return "Недостаточно данных для уверенного ответа на этот вопрос."

    posts_context = ""
    for i, post in enumerate(results, start=1):
        date = post.get("date")
        loc = post.get("location", "")
        text = post.get("text", "")
        img_desc = post.get("image_description", "")
        faces = post.get("faces", [])
        sentiment = post.get("sentiment", {})
        posts_context += f"{i}) "
        if date: posts_context += f"[{date}] "
        if loc: posts_context += f"{loc} "
        posts_context += text
        if img_desc: posts_context += f" (Описание фото: {img_desc})"
        # Добавим лица
        if faces:
            for face in faces:
                gender = face.get("gender")
                age = face.get("age")
                emo = face.get("dominant_emotion")
                posts_context += f" (На фото: {gender}, возраст {age}, эмоция {emo})"
        # Добавим сентимент
        if sentiment and isinstance(sentiment, dict) and "label" in sentiment:
            posts_context += f" (Настроение: {sentiment['label']})"
        posts_context += "\n"

    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt_template = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["personality"])
    prompt_body = prompt_template.format(date=current_date, posts=posts_context.strip())
    full_prompt = f"Вопрос: {question}\n\n" + prompt_body
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.3,
        max_tokens=512
    )
    answer = response.choices[0].message.content
    cache[key] = answer
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logging.info(f"question={question}, answer={answer}")
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
