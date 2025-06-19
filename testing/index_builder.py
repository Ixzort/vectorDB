import os
import json
import numpy as np
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CLOUD = "aws"
REGION = "us-east-1"
SPEC = ServerlessSpec(cloud=CLOUD, region=REGION)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def clean_metadata(meta):
    """Преобразует только вложенные dict и списки dict в строку, верхний уровень всегда dict!"""
    import numpy as np
    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            cleaned[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            # если список dict-ов — сериализуем каждый
            if any(isinstance(i, dict) for i in v):
                cleaned[k] = [json.dumps(i, ensure_ascii=False) if isinstance(i, dict) else i for i in v]
            else:
                cleaned[k] = [i if isinstance(i, (str, int, float, bool)) or i is None else str(i) for i in v]
        elif isinstance(v, np.generic):
            cleaned[k] = v.item()
        elif isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        elif isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned

def build_index(posts, index_name="social-index", max_posts: int = None):
    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=SPEC)
    idx = pc.Index(index_name)
    if max_posts:
        posts = posts[:max_posts]
    vecs = []
    for p in posts:
        if not p.get("text", "").strip() and not p.get("image_description", "").strip():
            continue
        embed_text = f"{p.get('text', '')} {p.get('image_description', '')}".strip()
        emb = openai_client.embeddings.create(input=embed_text, model="text-embedding-ada-002")
        meta = {k: v for k, v in p.items() if k != "image_url" and v is not None}
        meta = clean_metadata(meta)
        vecs.append({
            "id": p.get("post_id") or f"post_{len(vecs)}",
            "values": [float(x) for x in emb.data[0].embedding],
            "metadata": meta
        })
    print(f"\n🧠 Загружаем {len(vecs)} векторов в индекс «{index_name}»…")
    idx.upsert(vectors=vecs)
    print(f"✅ Всё загружено: {len(posts)} постов с текстом и image_description.")
    return posts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="processed_meta.json (уже обработанный)")
    parser.add_argument("-n", "--index", default="social-index")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    posts = json.load(open(args.input, encoding="utf-8"))
    build_index(posts, index_name=args.index, max_posts=args.max)
