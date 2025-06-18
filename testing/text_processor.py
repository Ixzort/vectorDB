import re, json, os, time
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def describe_image(url: str) -> str:
    if not url: return ""
    prompt = f"Опиши изображение по ссылке на русском в 1-2 предложениях (ссылка: {url})"
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,  # увеличено для длинных описаний!
            temperature=0.4
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Image description error for {url}: {e}")
        return ""

def preprocess_posts(posts: list[dict], max_posts: int = None) -> list[dict]:
    out = []
    if max_posts:
        posts = posts[:max_posts]
    for idx, post in enumerate(posts, 1):
        # Обычный текст поста
        text = (post.get("caption") or post.get("alt") or post.get("text") or
                post.get("firstComment") or post.get("ownerFullName") or post.get("ownerUsername") or "")

        # Описание картинки
        img_url = post.get("displayUrl") or post.get("image_url") or ""
        image_description = ""
        if img_url:
            image_description = describe_image(img_url)
            time.sleep(1.1)  # Не превышать лимиты OpenAI

        # Комментарии — список строк
        comments = []
        for c in post.get("latestComments", []):
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                comments.append(t.strip())

        out.append({
            "post_id": post.get("shortCode") or post.get("id") or "",
            "text": clean_text(text),
            "image_url": img_url,
            "image_description": image_description,
            "followers_count": post.get("followers_count", 0),
            "comments": comments,
            "date": post.get("date") or post.get("timestamp") or "",
            "ownerFullName": post.get("ownerFullName") or "",
            "ownerUsername": post.get("ownerUsername") or "",
        })
        print(f"[{idx}/{len(posts)}] Пост обработан. Длина text: {len(text)}. Длина описания фото: {len(image_description)}. Комментариев: {len(comments)}")
    return out

