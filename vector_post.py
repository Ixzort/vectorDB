import json, logging, os, random
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings  # <== ОБНОВЛЕНО
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Настройка OpenAI эмбеддинга
openai_api_key = os.getenv("OPENAI_API_KEY")
embed_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# CLIP-модель (ViT-B/32) и процессор для изображений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_url: str):
    """Возвращает 512-мерный CLIP-эмбеддинг для изображения по URL."""
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = clip_model.get_image_features(**inputs)
        return img_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        logging.error(f"Ошибка при получении эмбеддинга изображения {image_url}: {e}")
        return None

def process_posts(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    logging.info(f"Векторизация {len(posts)} постов")
    for post in posts:
        pid = post.get("id", "")
        # Объединяем caption + image_description для текстового эмбеддинга
        text = post.get("text", "")
        image_desc = post.get("metadata", {}).get("image_description", "")
        full_text = (text + ". " + image_desc).strip() if image_desc else text
        # Текстовый эмбеддинг
        try:
            vector_text = embed_model.embed_query(full_text)
            post["vector_text"] = vector_text
        except Exception as e:
            logging.error(f"Не удалось получить текстовый эмбеддинг для поста {pid}: {e}")
        # Эмбеддинг изображения
        img_url = post.get("image_url")
        if img_url:
            vector_image = get_image_embedding(img_url)
            post["vector_image"] = vector_image
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    logging.info("Сохранено JSON с эмбеддингами постов")

if __name__ == "__main__":
    input_path = "output_image.json"
    output_path = "output_vector.json"
    process_posts(input_path, output_path)
