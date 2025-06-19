import os
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def search_posts(query, index_name="social-index", top_k=5):
    idx = pc.Index(index_name)
    # можно комбинировать text и image_description для поиска
    emb = openai_client.embeddings.create(input=query, model="text-embedding-ada-002")
    vec = emb.data[0].embedding
    res = idx.query(vector=vec, top_k=top_k, include_metadata=True)
    results = []
    for match in res.get("matches", []):
        meta = match.get("metadata", {})
        results.append({
            "id": match.get("id"),
            "score": match.get("score"),
            "text": meta.get("text", ""),
            "image_description": meta.get("image_description", ""),
            "followers_count": meta.get("followers_count", 0),
            "comments": meta.get("comments", []),
            "date": meta.get("date", ""),
            "ownerFullName": meta.get("ownerFullName", ""),
            "ownerUsername": meta.get("ownerUsername", "")
        })
    return results

def get_posts_by_month(index_name, year_month, top_k=5):
    # year_month - строка "YYYY-MM"
    idx = pc.Index(index_name)
    all_res = idx.fetch()  # получите все посты (или иначе, как поддерживает ваш бэкенд)
    results = []
    for match in all_res.get("vectors", {}).values():
        meta = match.get("metadata", {})
        date = meta.get("date", "")
        if date.startswith(year_month):
            results.append({
                "id": match.get("id"),
                "score": None,
                "text": meta.get("text", ""),
                "image_description": meta.get("image_description", ""),
                "followers_count": meta.get("followers_count", 0),
                "comments": meta.get("comments", []),
                "date": meta.get("date", ""),
                "ownerFullName": meta.get("ownerFullName", ""),
                "ownerUsername": meta.get("ownerUsername", ""),
                "location": meta.get("location", "")
            })
            if len(results) >= top_k:
                break
    return results
