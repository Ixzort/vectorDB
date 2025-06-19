import re, json, os, time, logging
from typing import List
from deepface import DeepFace
from transformers import pipeline
import spacy
import numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)

# Список хостов или паттернов, которые нужно пропускать (можно расширить)
BAD_IMAGE_HOSTS = [
    "instagram.fosu2-1.fna.fbcdn.net",
    "fbcdn.net",
    "instagram.cdn",
    # Добавь свои паттерны если нужно
]

# Загрузка моделей
try:
    nlp = spacy.load("ru_core_news_lg")
except Exception:
    raise RuntimeError("spaCy ru_core_news_lg не установлена! pip install spacy && python -m spacy download ru_core_news_lg")

sentiment_pipeline = pipeline("sentiment-analysis", model="sismetanin/rubert-ru-sentiment-rusentiment")

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def download_image(img_url: str, save_path: str) -> str:
    try:
        import requests
        # Пропуск по "плохим" хостам
        if any(host in img_url for host in BAD_IMAGE_HOSTS):
            logging.warning(f"Пропускаю неподдерживаемую ссылку: {img_url}")
            return ""
        resp = requests.get(img_url, timeout=5)
        if resp.status_code != 200:
            logging.warning(f"Ошибка загрузки (код {resp.status_code}) {img_url}")
            return ""
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path
    except Exception as e:
        logging.warning(f"Ошибка скачивания {img_url}: {e}")
        return ""

def analyze_faces(img_path: str):
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        if isinstance(result, dict) and "age" in result:  # одиночное лицо
            result = [result]
        faces = []
        for face in result:
            faces.append({
                "age": face.get("age"),
                "gender": face.get("gender"),
                "dominant_emotion": face.get("dominant_emotion"),
                "emotion_scores": face.get("emotion", {}),
            })
        return faces
    except Exception as e:
        logging.warning(f"Ошибка распознавания лица на {img_path}: {e}")
        return []

def preprocess_posts(posts: List[dict], max_posts: int = None) -> List[dict]:
    out = []
    if max_posts:
        posts = posts[:max_posts]
    for idx, post in enumerate(posts, 1):
        text = (post.get("caption") or post.get("alt") or post.get("text") or
                post.get("firstComment") or post.get("ownerFullName") or post.get("ownerUsername") or "")
        img_url = post.get("displayUrl") or post.get("image_url") or ""
        image_description = ""
        local_path = ""
        faces = []
        if img_url:
            local_path = f"tmp_img_{idx}.jpg"
            img_path = download_image(img_url, local_path)
            faces = analyze_faces(img_path) if img_path else []
            try:
                if img_path:
                    os.remove(local_path)
            except Exception:
                pass
            time.sleep(1.1)
        # Sentiment
        sentiment = sentiment_pipeline(text)[0] if text.strip() else {}
        # NER
        doc = nlp(clean_text(text))
        persons = [ent.text for ent in doc.ents if ent.label_ == "PER"]
        locations = [ent.text for ent in doc.ents if ent.label_ in ("LOC", "GPE")]
        # Mentions
        mentions = re.findall(r'@(\w+)', text)
        # Comments
        comments = []
        for c in post.get("latestComments", []):
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                comments.append(t.strip())
        loc_name = post.get("locationName") or post.get("location") or ""
        raw_date = post.get("date") or post.get("timestamp") or ""
        date = raw_date.split("T")[0] if "T" in raw_date else raw_date
        out.append({
            "post_id": post.get("shortCode") or post.get("id") or "",
            "text": clean_text(text),
            "image_url": img_url,
            "faces": faces,
            "sentiment": sentiment,
            "persons": persons,
            "locations": locations,
            "mentions": mentions,
            "image_description": image_description,
            "followers_count": post.get("followers_count", 0),
            "comments": comments,
            "date": date,
            "location": loc_name,
            "ownerFullName": post.get("ownerFullName") or "",
            "ownerUsername": post.get("ownerUsername") or "",
        })
        print(f"[{idx}/{len(posts)}] Пост обработан. Лиц на фото: {len(faces)}. Sentiment: {sentiment.get('label')}. Persons: {persons}.")
    return out

# Для json — конвертация numpy float32 и других объектов
def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="processed_meta.json")
    args = parser.parse_args()
    posts = json.load(open(args.input, encoding="utf-8"))
    processed = preprocess_posts(posts)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2, default=to_serializable)
    print("Все посты обработаны и сохранены.")

