import json, re, logging, os, random
from datetime import datetime
import numpy as np
import spacy
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
random.seed(42)
np.random.seed(42)

nlp = spacy.load("xx_ent_wiki_sm")
sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    text = re.sub(r"&\w+;", "", text)
    return text.strip()

def extract_date_and_timestamp(post):
    # Поиск времени: сначала timestamp (в секундах или миллисекундах), затем ISO-строка
    raw = (
        post.get("timestamp")
        or post.get("date")
        or post.get("taken_at")
        or post.get("created_time")
        or post.get("created_at")
        or None
    )
    date_iso = None
    timestamp = None
    if raw is not None:
        # Если это ISO-строка
        if isinstance(raw, str) and ("T" in raw or "-" in raw):
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                date_iso = dt.date().isoformat()
                timestamp = int(dt.timestamp())
            except Exception:
                # Может быть всё-таки timestamp как строка
                try:
                    ts = int(raw)
                    # Если timestamp в миллисекундах — приведём к секундам
                    if ts > 1e12:
                        ts = ts // 1000
                    date_iso = datetime.utcfromtimestamp(ts).date().isoformat()
                    timestamp = ts
                except Exception:
                    date_iso = None
                    timestamp = None
        else:
            # Попробуем как int
            try:
                ts = int(raw)
                if ts > 1e12:
                    ts = ts // 1000
                date_iso = datetime.utcfromtimestamp(ts).date().isoformat()
                timestamp = ts
            except Exception:
                date_iso = None
                timestamp = None
    return date_iso, timestamp

def process_posts(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        posts_data = json.load(f)
    processed = []
    logging.info(f"Загружено {len(posts_data)} постов")
    categories = {
        "спорт": ["gym", "workout", "run", "sport", "тренировка", "фитнес"],
        "путешествия": ["travel", "flight", "trip", "поездка", "отпуск", "тур"],
        "еда": ["restaurant", "recipe", "dinner", "ресторан", "рецепт", "ужин"],
    }
    for post in posts_data:
        pid = post.get("shortCode") or post.get("id") or ""
        logging.info(f"Обработка поста {pid}")
        caption = post.get("caption", "") or ""
        text = clean_text(caption)
        logging.debug(f"Очищенный текст: {text[:30]}...")
        # --- Новая обработка времени ---
        date_iso, timestamp = extract_date_and_timestamp(post)
        hashtags = re.findall(r"#(\w+)", text)
        mentions = re.findall(r"@(\w+)", text)
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "EVENT")]
        location = post.get("locationName")
        if not location:
            loc_entities = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
            location = loc_entities[0] if loc_entities else None
        activity = None
        lower_text = text.lower()
        for cat, keywords in categories.items():
            if any(kw in lower_text for kw in keywords):
                activity = cat
                break
        sentiment = None
        if text:
            try:
                result = sentiment_pipe(text[:512])[0]
                sentiment = result.get("label")
            except Exception as e:
                logging.error(f"Ошибка анализа сентимента поста {pid}: {e}")
        processed_post = {
            "id": pid,
            "text": text,
            "image_url": post.get("displayUrl") or post.get("image_url"),
            "metadata": {
                "date": date_iso,
                "timestamp": timestamp,
                "hashtags": hashtags,
                "mentions": mentions,
                "entities": entities,
                "location": location,
                "activity_type": activity,
                "sentiment": sentiment,
            }
        }
        processed.append(processed_post)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    logging.info(f"Сохранено очищенных постов: {len(processed)}")

if __name__ == "__main__":
    input_path = "export.json"
    output_path = "output_clear.json"
    process_posts(input_path, output_path)

