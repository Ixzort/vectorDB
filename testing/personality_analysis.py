traits = {
    "экстраверсия": ["весёлый", "общительный", "дружелюбный", "вечеринка", "активный"],
    "добросовестность": ["ответственный", "аккуратный", "организованный", "дисциплинированный"],
    "нейротизм": ["тревожный", "нервный", "переживает", "печальный", "раздражительный"],
    "открытость": ["любопытный", "открытый", "новое", "исследование", "творческий"],
    "доброжелательность": ["доброжелательный", "отзывчивый", "сострадательный", "щедрый", "помогающий"]
}

import json

with open("processed_meta.json", encoding="utf-8") as f:
    posts = json.load(f)

scores = dict.fromkeys(traits.keys(), 0)
texts = [p["text"] for p in posts]

for text in texts:
    t = text.lower()
    for trait, keywords in traits.items():
        for word in keywords:
            if word in t:
                scores[trait] += 1

total = sum(scores.values())
print("Big Five scores:")
for trait, sc in scores.items():
    print(f"{trait}: {sc} ({(sc / total * 100) if total else 0:.1f}%)")
