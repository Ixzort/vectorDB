import json, logging, os
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def describe_image_with_vision(image_url: str) -> str:
    if not image_url:
        return None
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Опиши это фото:"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.0
        )
        desc = response.choices[0].message.content.strip()
        return desc
    except Exception as e:
        logging.error(f"Не удалось получить описание изображения {image_url}: {e}")
        return None

def process_posts(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    logging.info(f"Загружено {len(posts)} постов для анализа изображений")
    for post in posts:
        pid = post.get("id", "")
        img_url = post.get("image_url")
        if img_url:
            logging.info(f"Получение описания для поста {pid}")
            description = describe_image_with_vision(img_url)
            if description:
                post.setdefault("metadata", {})["image_description"] = description
                logging.debug(f"Описание: {description[:50]}...")
        else:
            logging.info(f"Пост {pid} не содержит изображения")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    logging.info("Сохранено описание изображений в JSON")

if __name__ == "__main__":
    input_path = "output_clear.json"
    output_path = "output_image.json"
    process_posts(input_path, output_path)


