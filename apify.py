import os
import json
import logging
from apify_client import ApifyClient
from typing import List, Dict, Union

# Логирование Apify SDK
logging.basicConfig(level=logging.INFO)
logging.getLogger('apify_client').setLevel(logging.DEBUG)

class InstagramScraperService:
    def __init__(self,
                 output_dir: str = 'inst_data',
                 output_filename: str = 'export.json',
                 wait_secs: int = 300):
        self.token = os.getenv('APIFY_TOKEN')
        print("🔑 APIFY_TOKEN:", "SET" if self.token else "NOT SET")
        if not self.token:
            raise ValueError("APIFY_TOKEN не задан!")
        self.client = ApifyClient(self.token)
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, output_filename)
        self.wait_secs = wait_secs

    def fetch_posts(self, usernames: Union[str, List[str]], limit: int = 10) -> List[Dict]:
        if isinstance(usernames, str):
            usernames = [usernames]
        urls = [f"https://www.instagram.com/{u.strip().lstrip('@')}/" for u in usernames]
        print("🔍 Target URLs:", urls)

        run_input = {
            "directUrls": urls,
            "resultsType": "posts",
            "resultsLimit": limit
        }
        print("📤 Input to actor:", run_input)

        try:
            run = self.client.actor("apify/instagram-scraper").call(
                run_input=run_input,
                wait_secs=self.wait_secs  # WAIT правильно указано
            )
            print("✅ Actor finished:", run.get("status"), run.get("statusMessage"))

            ds_id = run.get("defaultDatasetId")
            print("🗄 Dataset ID:", ds_id)
            if not ds_id:
                print("❗ defaultDatasetId отсутствует — актер мог не сохранить данные.")
                return []

            items = self.client.dataset(ds_id).list_items().items
            print(f"✅ Получено {len(items)} постов")

            if items:
                with open(self.output_path, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
                print("💾 Сохранено в файл:", self.output_path)
            else:
                print("⚠️ Датасет пуст — возможно профиль приватен или нет новых постов.")

            return items

        except Exception as e:
            print("❌ Ошибка выполнения актора:", repr(e))
            return []

# Пример вызова
if __name__ == "__main__":
    svc = InstagramScraperService()
    posts = svc.fetch_posts(['daniella.ssi'], limit=40)
    print("🔚 Всего постов:", len(posts))

