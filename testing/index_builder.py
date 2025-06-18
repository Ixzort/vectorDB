import os, json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from text_processor import preprocess_posts

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
CLOUD = "aws"
REGION = "us-east-1"
SPEC = ServerlessSpec(cloud=CLOUD, region=REGION)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def build_index(posts, index_name="social-index", max_posts: int = None):
    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=SPEC)
    idx = pc.Index(index_name)
    processed = preprocess_posts(posts, max_posts=max_posts)
    vecs = []
    for p in processed:
        if not p["text"].strip() and not p["image_description"].strip():
            continue
        # для embedding лучше комбинировать text и image_description
        embed_text = f"{p['text']} {p['image_description']}".strip()
        emb = openai_client.embeddings.create(input=embed_text, model="text-embedding-ada-002")
        meta = {k: v for k, v in p.items() if k != "image_url" and v is not None}
        # делаем сериализацию comments для Pinecone, если нужен поиск по ним
        if isinstance(meta.get("comments"), list):
            meta["comments"] = [str(x) for x in meta["comments"]]
        vecs.append({
            "id": p["post_id"] or f"post_{len(vecs)}",
            "values": emb.data[0].embedding,
            "metadata": meta
        })
    print(f"\n🧠 Загружаем {len(vecs)} векторов в индекс «{index_name}»…")
    idx.upsert(vectors=vecs)
    with open("processed_meta.json", "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"✅ Всё загружено: {len(processed)} постов с текстом и image_description.")
    return processed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-n", "--index", default="social-index")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    posts = json.load(open(args.input, encoding="utf-8"))
    processed = build_index(posts, index_name=args.index, max_posts=args.max)


