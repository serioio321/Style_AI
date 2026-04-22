import base64
import json
from pathlib import Path
from typing import Optional, List

import ollama

SYSTEM_PROMPT = """Ты — эксперт по моде и стилю. Анализируй изображение и возвращай JSON.

Первым делом определи: это вещь из гардероба (одежда, обувь, аксессуар)?

Если НЕТ — верни: {"is_clothing": false}

Если ДА — верни все поля:
- is_clothing: true
- category: тип вещи (футболка, джинсы, платье, пальто, кроссовки, сумка и т.д.)
- color: основной цвет (белый, чёрный, синий, красный, бежевый и т.д.)
- style: стиль (casual, formal, sport, streetwear, elegant, bohemian)
- season: сезон (лето, зима, демисезон, всесезонный)
- description: краткое описание на русском (1-2 предложения)
- tags: список из 3-5 тегов (например ["базовый", "однотонный", "универсальный"])

Отвечай ТОЛЬКО валидным JSON без markdown-блоков."""

SHOP_CATEGORIES = {
    "верх": ["футболка", "рубашка", "блузка", "свитер", "худи", "толстовка", "джемпер", "топ"],
    "низ": ["джинсы", "брюки", "шорты", "юбка", "леггинсы"],
    "верхняя одежда": ["пальто", "куртка", "пуховик", "тренч", "плащ", "ветровка"],
    "платья и комбинезоны": ["платье", "сарафан", "комбинезон", "костюм"],
    "обувь": ["кроссовки", "туфли", "ботинки", "сапоги", "сандалии", "лоферы", "мокасины"],
    "аксессуары": ["сумка", "ремень", "шарф", "шапка", "перчатки", "украшения"],
}

MISSING_CATEGORIES_TIPS = {
    "верх": "Добавь базовую футболку или рубашку — они сделают образы более универсальными",
    "низ": "В гардеробе не хватает брюк или джинсов — основы большинства образов",
    "верхняя одежда": "Нет верхней одежды — пальто или куртка завершат любой образ",
    "платья и комбинезоны": "Платье или костюм добавят возможности для формальных образов",
    "обувь": "Без обуви образ не завершить — кроссовки или туфли must-have",
    "аксессуары": "Аксессуары (сумка, ремень) поднимают образ на новый уровень",
}


async def classify_image(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    client = ollama.AsyncClient()
    response = await client.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": SYSTEM_PROMPT + "\n\nКлассифицируй эту вещь. Отвечай ТОЛЬКО валидным JSON.",
                "images": [image_data],
            }
        ],
    )

    raw = response.message.content.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def get_category_group(category: str) -> Optional[str]:
    category_lower = category.lower()
    for group, items in SHOP_CATEGORIES.items():
        if any(item in category_lower for item in items):
            return group
    return None


def analyze_gaps(items: List[dict]) -> List[str]:
    present_groups = set()
    for item in items:
        group = get_category_group(item.get("category", ""))
        if group:
            present_groups.add(group)

    tips = []
    for group, tip in MISSING_CATEGORIES_TIPS.items():
        if group not in present_groups:
            tips.append(tip)

    return tips
