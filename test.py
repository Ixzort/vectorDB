import json, re
from datetime import datetime
import spacy
from transformers import pipeline

nlp = spacy.load("xx_ent_wiki_sm")
sentiment_pipe = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)

emoji_re = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
html_re = re.compile(r"<.*?>")

def clean_text(text: str) -> str:
    t = html_re.sub(" ", text or "")
    t = emoji_re.sub("", t)
    t = re.sub(r"[^\w\s#@ёЁйЙа-яА-Я0-9.,!?…]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

with open("export.json", encoding="utf-8") as f:
    raw_posts = json.load(f)

processed = []
for p in raw_posts:
    text = clean_text(p.get("caption", ""))

    date_iso = None
    for fld in ("timestamp", "date"):
        raw = p.get(fld)
        if raw:
            try:
                date_iso = datetime.fromisoformat(raw.rstrip("Z")).strftime("%Y-%m-%d")
                break
            except:
                pass

    hashtags = p.get("hashtags") or re.findall(r"#\w+", text)
    mentions = p.get("mentions", [])

    image_url = p.get("displayUrl") or p.get("image_url") or p.get("url")
    img_path = image_url

    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    locs = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
    location = locs[0] if locs else None

    sentiment = sentiment_pipe(text[:512])[0]["label"] if text else None

    cats = {
        "travel": ["travel", "trip", "beach", "flight"],
        "food": ["recipe", "restaurant", "dinner", "ужин", "рецепт"],
        "sport": ["gym", "fitness", "run", "тренировка"]
    }
    activity = None
    lc = text.lower()
    for cat, keys in cats.items():
        if any(k in lc for k in keys) or any(f"#{k}" in hashtags for k in keys):
            activity = cat
            break

    processed.append({
        "id": p.get("id"),
        "short_code": p.get("shortCode"),
        "type": p.get("type"),
        "text": text,
        "image_path": img_path,
        "metadata": {
            "date": date_iso,
            "hashtags": hashtags,
            "mentions": mentions,
            "location": location,
            "entities": entities,
            "sentiment": sentiment,
            "activity_type": activity
        }
    })

with open("processed.json", "w", encoding="utf-8") as out:
    json.dump(processed, out, ensure_ascii=False, indent=2)

print(f"✅ Обработано {len(processed)} постов.")
