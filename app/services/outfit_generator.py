from typing import List, Dict, Optional
import ollama
from transformers import MarianMTModel, MarianTokenizer

_translator_model = None
_translator_tokenizer = None


def _load_translator():
    global _translator_model, _translator_tokenizer
    if _translator_model is None:
        print("Загружаю модель перевода en→ru...")
        name = "Helsinki-NLP/opus-mt-en-ru"
        _translator_tokenizer = MarianTokenizer.from_pretrained(name)
        _translator_model = MarianMTModel.from_pretrained(name)
        print("Модель перевода загружена.")
    return _translator_model, _translator_tokenizer


def _translate_en_ru(text: str) -> str:
    model, tokenizer = _load_translator()
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

STYLE_RU = {
    "casual": "кэжуал", "formal": "официальный", "sport": "спортивный",
    "streetwear": "уличный", "elegant": "элегантный", "bohemian": "богемный",
}

def normalize_style(style: str) -> str:
    return STYLE_RU.get(style.lower(), style)

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

TOP_CATEGORIES = ["футболка", "лонгслив", "поло", "рубашка", "блузка", "топ", "боди",
                  "свитер", "худи", "толстовка", "джемпер", "кардиган", "водолазка",
                  "жилет", "пиджак"]
BOTTOM_CATEGORIES = ["джинсы", "брюки", "чиносы", "шорты", "бермуды", "юбка",
                     "мини-юбка", "леггинсы", "спортивные штаны"]
OUTER_CATEGORIES = ["пальто", "куртка", "пуховик", "тренч", "плащ",
                    "ветровка", "бомбер", "парка", "дублёнка"]
DRESS_CATEGORIES = ["платье", "вечернее платье", "сарафан", "комбинезон", "костюм"]
SHOES_CATEGORIES = ["кроссовки", "туфли", "ботинки", "сапоги", "сандалии",
                    "шлёпанцы", "лоферы", "мокасины", "балетки", "угги"]
ACC_CATEGORIES = ["сумка", "рюкзак", "клатч", "ремень", "шарф", "шапка",
                  "кепка", "панама", "перчатки", "украшения", "очки", "галстук", "часы"]


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
    styles = [normalize_style(item.get("style", "")).lower() for item in items]

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
                 "category": i.get("category"), "color": i.get("color"),
                 "style": normalize_style(i.get("style", ""))}
                for i in combo
            ],
            "score": round(score, 2),
            "style": normalize_style(combo[0].get("style", "кэжуал")),
        })
        if len(outfits) >= max_outfits:
            break

    return outfits


_NEUTRALS = {"чёрный", "белый", "серый", "бежевый", "коричневый"}

_STYLE_OCCASIONS = {
    "официальный": ["Офис", "Деловая встреча", "Презентация"],
    "элегантный":  ["Ужин", "Свидание", "Торжество"],
    "кэжуал":      ["Прогулка", "Встреча с друзьями", "Повседневно"],
    "спортивный":  ["Тренировка", "Активный отдых", "Фитнес"],
    "уличный":     ["Прогулка по городу", "Концерт", "Тусовка"],
    "богемный":    ["Выставка", "Кафе", "Арт-событие"],
}


def _quality(score: float) -> tuple:
    if score >= 3:
        return "Отличное сочетание", "great"
    if score >= 1:
        return "Хороший образ", "good"
    return "Смелое сочетание", "mixed"


def _occasions(style: str) -> list:
    return _STYLE_OCCASIONS.get(style, ["Повседневно"])


def _color_insight(colors: list) -> str:
    unique = list(dict.fromkeys(c for c in colors if c))
    if not unique:
        return ""

    if len(unique) == 1:
        return f"Монохромный образ в {unique[0]} тонах — лаконично и стильно."

    if len(unique) == 2:
        c1, c2 = unique[0], unique[1]
        both_neutral = c1 in _NEUTRALS and c2 in _NEUTRALS
        compatible = c2 in COLOR_COMPATIBILITY.get(c1, []) or c1 in COLOR_COMPATIBILITY.get(c2, [])
        if both_neutral:
            return f"Нейтральная гамма {c1} + {c2} — беспроигрышный вариант."
        if compatible:
            return f"Гармоничное сочетание {c1} и {c2}."
        return f"Контрастный образ: {c1} против {c2} — смелый выбор."

    brights = [c for c in unique if c not in _NEUTRALS]
    if not brights:
        return "Полностью нейтральная гамма — легко сочетается с чем угодно."
    if len(brights) == 1:
        return f"Яркий акцент {brights[0]} на нейтральной базе."
    return f"Насыщенная палитра: {', '.join(unique[:3])}."


def _style_tip(items: list) -> Optional[str]:
    cats = {i.get("category", "").lower() for i in items}
    styles = {normalize_style(i.get("style", "")) for i in items}

    if "спортивные штаны" in cats and styles & {"официальный", "элегантный"}:
        return "Спортивные штаны не сочетаются с формальными вещами — попробуй джинсы или брюки."
    if "леггинсы" in cats and styles & {"официальный", "элегантный"}:
        return "Леггинсы лучше сочетать с кэжуал вещами, не с официальными."
    if "шлёпанцы" in cats and styles & {"официальный", "элегантный"}:
        return "Шлёпанцы слишком расслаблены для этого образа — рассмотри ботинки или туфли."
    if {"официальный", "спортивный"} <= styles:
        sport_item = next((i for i in items if normalize_style(i.get("style", "")) == "спортивный"), None)
        if sport_item:
            return f"{sport_item['category'].capitalize()} выбивается из формального образа — замени на более строгую альтернативу."
    return None


def analyze_outfit(outfit: dict) -> dict:
    items = outfit["items"]
    score = outfit.get("score", 0)
    style = normalize_style(outfit.get("style", "кэжуал"))

    quality, quality_level = _quality(score)
    occasions = _occasions(style)
    insight = _color_insight([i.get("color", "") for i in items])
    tip = _style_tip(items)

    return {
        "quality": quality,
        "quality_level": quality_level,
        "occasions": occasions,
        "insight": insight,
        "tip": tip,
    }


async def describe_outfit(outfit: dict) -> str:
    items = outfit["items"]
    style = normalize_style(outfit.get("style", "кэжуал"))
    items_en = ", ".join(f"{i['category']} in {i['color']}" for i in items)

    try:
        # Step 1: English description via llama3.2
        client = ollama.AsyncClient()
        response = await client.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are a fashion stylist. Reply in English only."},
                {"role": "user", "content": f"Outfit: {items_en}. Style: {style}. Complete this sentence in 6-8 words: 'Perfect for...' — name 1-2 real occasions. No greetings, no exclamations."},
            ],
        )
        en_text = response.message.content.strip()
        for sep in ".!?":
            idx = en_text.find(sep)
            if idx != -1:
                en_text = en_text[:idx + 1]
                break
        en_text = en_text[:300]

        # Step 2: translate offline via Helsinki-NLP/opus-mt-en-ru
        import asyncio
        loop = asyncio.get_event_loop()
        ru_text = await loop.run_in_executor(None, _translate_en_ru, en_text)
        return ru_text.strip()
    except Exception as e:
        print(f"[describe_outfit error] {e}")
        return ""
