from typing import List, Dict, Optional
import ollama

COLOR_COMPATIBILITY = {
    "чёрный": ["белый", "серый", "бежевый", "красный", "синий", "зелёный", "розовый", "жёлтый"],
    "белый": ["чёрный", "серый", "синий", "бежевый", "красный", "зелёный", "розовый"],
    "серый": ["чёрный", "белый", "синий", "бордовый", "розовый", "бежевый"],
    "синий": ["белый", "серый", "бежевый", "чёрный", "голубой"],
    "бежевый": ["чёрный", "белый", "коричневый", "синий", "серый", "зелёный"],
    "коричневый": ["бежевый", "белый", "оранжевый", "зелёный", "кремовый"],
    "зелёный": ["бежевый", "белый", "коричневый", "чёрный"],
    "красный": ["чёрный", "белый", "серый", "синий"],
    "розовый": ["серый", "белый", "чёрный", "бежевый"],
    "бордовый": ["серый", "белый", "бежевый", "чёрный"],
}

TOP_CATEGORIES = ["футболка", "рубашка", "блузка", "свитер", "худи", "толстовка", "джемпер", "топ"]
BOTTOM_CATEGORIES = ["джинсы", "брюки", "шорты", "юбка", "леггинсы"]
OUTER_CATEGORIES = ["пальто", "куртка", "пуховик", "тренч", "плащ", "ветровка"]
DRESS_CATEGORIES = ["платье", "сарафан", "комбинезон", "костюм"]
SHOES_CATEGORIES = ["кроссовки", "туфли", "ботинки", "сапоги", "сандалии", "лоферы", "мокасины"]
ACC_CATEGORIES = ["сумка", "ремень", "шарф", "шапка", "перчатки"]


def _get_slot(category: str) -> str:
    c = category.lower()
    if any(x in c for x in TOP_CATEGORIES):
        return "верх"
    if any(x in c for x in BOTTOM_CATEGORIES):
        return "низ"
    if any(x in c for x in OUTER_CATEGORIES):
        return "верхняя одежда"
    if any(x in c for x in DRESS_CATEGORIES):
        return "платье"
    if any(x in c for x in SHOES_CATEGORIES):
        return "обувь"
    if any(x in c for x in ACC_CATEGORIES):
        return "аксессуары"
    return "другое"


def _colors_match(color1: str, color2: str) -> bool:
    c1 = color1.lower()
    c2 = color2.lower()
    return c2 in COLOR_COMPATIBILITY.get(c1, []) or c1 == c2


def _score_combination(items: List[dict]) -> float:
    score = 0.0
    colors = [item.get("color", "").lower() for item in items]
    styles = [item.get("style", "").lower() for item in items]

    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if _colors_match(colors[i], colors[j]):
                score += 1.0

    if len(set(styles)) == 1:
        score += 2.0
    elif len(set(styles)) <= 2:
        score += 1.0

    return score


def generate_outfits(items: List[dict], max_outfits: int = 3) -> List[dict]:
    by_slot: Dict[str, List[dict]] = {}
    for item in items:
        slot = _get_slot(item.get("category", ""))
        by_slot.setdefault(slot, []).append(item)

    tops = by_slot.get("верх", []) + by_slot.get("платье", [])
    bottoms = by_slot.get("низ", [])
    shoes = by_slot.get("обувь", [])
    outers = by_slot.get("верхняя одежда", [None])
    accessories = by_slot.get("аксессуары", [None])

    candidates = []
    for top in tops:
        for bottom in (bottoms or [None]):
            if top.get("category", "").lower() in DRESS_CATEGORIES and bottom:
                continue
            for shoe in (shoes or [None]):
                for outer in outers:
                    for acc in accessories:
                        combo = [x for x in [top, bottom, shoe, outer, acc] if x]
                        if len(combo) >= 2:
                            candidates.append((_score_combination(combo), combo))

    candidates.sort(key=lambda x: x[0], reverse=True)

    outfits = []
    seen_tops = set()
    for score, combo in candidates:
        top_id = combo[0].get("id")
        if top_id in seen_tops:
            continue
        seen_tops.add(top_id)
        outfits.append({
            "items": [
                {"id": i.get("id"), "filename": i.get("filename"),
                 "category": i.get("category"), "color": i.get("color"), "style": i.get("style")}
                for i in combo
            ],
            "score": round(score, 2),
            "style": combo[0].get("style", "casual"),
        })
        if len(outfits) >= max_outfits:
            break

    return outfits


async def describe_outfit(outfit: dict) -> str:
    items_desc = ", ".join(f"{i['category']} ({i['color']})" for i in outfit["items"])

    client = ollama.AsyncClient()
    response = await client.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": f"Опиши образ из этих вещей одним-двумя предложениями на русском, как стилист: {items_desc}. "
                           f"Укажи когда и куда подойдёт этот образ.",
            }
        ],
    )
    return response.message.content.strip()
