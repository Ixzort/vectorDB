import json, os, time
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "social-index"
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def describe_image(url: str) -> str:
    if not url:
        return ""
    prompt = f"Опиши изображение по ссылке на русском в 1-2 предложениях (ссылка: {url})"
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.4
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Image description error for {url}: {e}")
        return ""

if __name__ == "__main__":
    meta = json.load(open("processed_meta.json", encoding="utf-8"))
    idx = pc.Index(INDEX_NAME)
    updated = 0
    for p in meta:
        url = p.get("image_url")
        if url and not p.get("image_description"):
            desc = describe_image(url)
            p["image_description"] = desc
            # Обновляем только поле image_description (Pinecone позволяет патчить метаданные)
            idx.upsert([{
                "id": p["post_id"],
                "values": [],
                "metadata": {"image_description": desc}
            }], sparse=False)
            print(f"{p['post_id']}: {desc}")
            updated += 1
            time.sleep(1.1)  # чтобы не превышать лимит токенов/запросов
    json.dump(meta, open("processed_meta_with_imgdesc.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Описания изображений добавлены к {updated} постам.")
