import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

_executor = ThreadPoolExecutor(max_workers=2)
_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("Загружаю CLIP модель...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print("Модель загружена.")
    return _clip_model, _clip_processor


# --- Категории ---

CATEGORY_LABELS = {
    # Верх
    "футболка":       "a plain short sleeve cotton t-shirt, thin lightweight fabric, crew neck",
    "лонгслив":       "a long sleeve tight base layer top, thin fabric, no buttons",
    "поло":           "a polo shirt with collar and buttons at neck",
    "рубашка":        "a button-up dress shirt with collar and front buttons",
    "блузка":         "a women blouse with decorative details",
    "топ":            "a sleeveless tank top or crop top",
    "боди":           "a bodysuit that snaps at the bottom",
    "свитер":         "a thick knitted wool sweater, textured fabric",
    "худи":           "a hoodie sweatshirt with a hood attached",
    "толстовка":      "a thick fleece sweatshirt with no hood, heavy fabric",
    "джемпер":        "a knitted pullover jumper, medium weight",
    "кардиган":       "an open-front cardigan with buttons down the front",
    "водолазка":      "a turtleneck top with high neck that folds over",
    "жилет":          "a sleeveless vest or waistcoat",
    "пиджак":         "a structured blazer jacket with lapels",
    # Низ
    "джинсы":         "jeans denim pants",
    "брюки":          "formal trousers",
    "чиносы":         "chino pants",
    "шорты":          "shorts",
    "бермуды":        "bermuda shorts",
    "юбка":           "a skirt",
    "мини-юбка":      "a mini skirt",
    "леггинсы":       "leggings",
    "спортивные штаны": "sweatpants or joggers",
    # Верхняя одежда
    "пальто":         "a long wool coat",
    "куртка":         "a casual jacket",
    "пуховик":        "a puffer down jacket",
    "тренч":          "a trench coat",
    "плащ":           "a raincoat",
    "ветровка":       "a windbreaker",
    "бомбер":         "a bomber jacket",
    "парка":          "a parka",
    "дублёнка":       "a shearling coat",
    # Платья и комбинезоны
    "платье":         "a dress",
    "вечернее платье": "an evening gown",
    "сарафан":        "a sundress",
    "комбинезон":     "a jumpsuit or overalls",
    "костюм":         "a formal suit",
    # Обувь
    "кроссовки":      "sneakers or athletic shoes",
    "туфли":          "dress shoes or heels",
    "ботинки":        "ankle boots",
    "сапоги":         "tall boots",
    "сандалии":       "sandals",
    "шлёпанцы":       "flip flops or slides",
    "лоферы":         "loafers",
    "мокасины":       "moccasins",
    "балетки":        "ballet flats",
    "угги":           "ugg boots",
    # Аксессуары
    "сумка":          "a handbag",
    "рюкзак":         "a backpack",
    "клатч":          "a clutch bag",
    "ремень":         "a belt",
    "шарф":           "a scarf",
    "шапка":          "a beanie or winter hat",
    "кепка":          "a baseball cap",
    "панама":         "a bucket hat",
    "перчатки":       "gloves",
    "носки":          "socks",
    "украшения":      "jewelry accessories",
    "очки":           "sunglasses or eyeglasses",
    "галстук":        "a necktie",
    "часы":           "a wristwatch",
}

CLOTHING_CHECK_LABELS = [
    "a clothing item, shoe, or fashion accessory",
    "food, animal, landscape, furniture, electronics, or other non-clothing object",
]

# Этап 1 — широкие группы
GROUP_LABELS = {
    "верх":              "a garment worn only on the upper body: t-shirt, shirt, sweater, hoodie, blouse, does not cover the legs",
    "низ":               "a garment worn only on the lower body covering the legs or waist: pants, jeans, trousers, skirt, shorts",
    "верхняя одежда":    "an outer layer jacket or coat worn over other clothing for warmth, outerwear",
    "платье":            "a one-piece garment that covers both the torso and legs together: dress, gown, jumpsuit",
    "обувь":             "footwear worn on the feet: sneakers, boots, shoes, sandals, heels, slippers",
    "аксессуары":        "a wearable fashion accessory not covering the body: bag, belt, scarf, hat, jewelry, watch, sunglasses, gloves",
}

# Этап 2 — категории внутри каждой группы
GROUP_CATEGORIES = {
    "верх": {
        "футболка":   "a t-shirt: very short sleeves ending at the upper arm, simple round crew neck, no collar, no buttons, plain thin jersey fabric",
        "лонгслив":   "a long sleeve shirt: sleeves extend fully to the wrist, no collar, no buttons, fitted thin fabric, solid color base layer",
        "поло":       "a polo shirt: short sleeves, ribbed polo collar standing up, 2 or 3 buttons at the neck opening only",
        "рубашка":    "a dress shirt: formal spread collar, buttons running down the entire front placket, cuffs at the wrist",
        "блузка":     "a women's blouse: lightweight decorative fabric, feminine cut, may have ruffles, bow, or floral print",
        "топ":        "a sleeveless top or crop top: no sleeves at all, bare shoulders, short length, tank or crop style",
        "боди":       "a bodysuit: like a t-shirt but extends to the hips, snaps or buttons at the crotch, no separate bottom",
        "свитер":     "a knitted sweater: visibly knitted ribbed texture, wool or acrylic yarn, thick warm fabric, chunky cables or ribs",
        "худи":       "a hoodie: thick sweatshirt with a large hood attached at the back of the neck, kangaroo pocket in front",
        "толстовка":  "a crewneck sweatshirt: thick heavy fleece fabric, NO hood, round neck, ribbed cuffs, heavier than a t-shirt",
        "джемпер":    "a pullover jumper: smooth knit fabric, no hood, no buttons, no collar, medium weight, pulled over the head",
        "кардиган":   "a cardigan: knitted sweater that is fully open at the front with buttons or zipper all the way down",
        "водолазка":  "a turtleneck: high tubular neck that folds or rolls over, covers the chin area, long sleeves",
        "жилет":      "a vest or waistcoat: no sleeves, covers only the torso, either knitted or structured with buttons",
        "пиджак":     "a blazer: structured jacket with wide lapels, inside lining, formal cut, buttons at front, worn over other tops",
    },
    "низ": {
        "джинсы":            "jeans: blue or dark denim fabric with visible stitching, rivets at pockets, denim weave texture",
        "брюки":             "formal trousers: smooth pressed fabric, pleated or flat front, dress pants for office",
        "чиносы":            "chino pants: beige or khaki cotton twill fabric, casual smart look, not denim",
        "шорты":             "shorts: leg opening ends well above the knee, warm weather bottoms",
        "бермуды":           "bermuda shorts: leg ends just below the knee, longer than regular shorts",
        "юбка":              "a skirt: midi or maxi length, falls below the knee, no legs, flowy fabric",
        "мини-юбка":         "a mini skirt: very short, ends high on the thigh, well above the knee",
        "леггинсы":          "leggings: extremely tight stretch fabric, covers full leg to ankle, worn as athletic or casual bottoms",
        "спортивные штаны":  "sweatpants: loose or tapered leg, elastic waistband, soft fleece inside, jogger style",
    },
    "верхняя одежда": {
        "пальто":    "a long coat: reaches the knee or below, wool or heavy fabric, formal outerwear",
        "куртка":    "a casual jacket: ends at the hip, casual style, zipper or buttons, not quilted",
        "пуховик":   "a puffer jacket: visibly quilted diamond or horizontal stitching pattern, filled with down insulation, puffy appearance",
        "тренч":     "a trench coat: double-breasted buttons, fabric belt at waist, shoulder epaulettes, mid-length",
        "плащ":      "a raincoat: waterproof shiny fabric, often with a hood, rain protection",
        "ветровка":  "a windbreaker: very lightweight thin nylon or polyester shell, no heavy insulation, packable",
        "бомбер":    "a bomber jacket: short length, ribbed elastic band at hem and cuffs, zipper front, MA-1 style",
        "парка":     "a parka: long heavy insulated jacket with a hood, fur-trimmed hood edge, very warm winter coat",
        "дублёнка":  "a shearling coat: genuine sheepskin or faux fur inside visible at collar and cuffs, fuzzy warm lining",
    },
    "платье": {
        "платье":          "a casual dress: one-piece garment covering torso and legs, everyday wear, various sleeve lengths",
        "вечернее платье": "an evening gown or cocktail dress: formal elegant dress, floor length or knee length, special occasion",
        "сарафан":         "a sundress: sleeveless dress with thin shoulder straps, casual summer style",
        "комбинезон":      "a jumpsuit or overalls: one-piece with both top and pants combined, legs clearly visible",
        "костюм":          "a matching suit set: jacket and trousers or skirt in the same fabric and color, formal coordinated set",
    },
    "обувь": {
        "кроссовки":  "sneakers: athletic running or lifestyle shoes with rubber sole, laces, chunky sporty silhouette",
        "туфли":      "dress shoes or heels: formal leather shoes or high heel pumps, pointed or round toe, no laces",
        "ботинки":    "ankle boots: cover the ankle, rigid sole, stop just above the ankle bone",
        "сапоги":     "tall boots: shaft rises to the knee or higher, covers the calf",
        "сандалии":   "sandals: open-toe, straps across the foot, no enclosed toe box, summer footwear",
        "шлёпанцы":   "flip flops or slides: completely flat sole, minimal straps, worn without socks, pool shoes",
        "лоферы":     "loafers: slip-on shoes without laces, low heel, classic casual or smart casual",
        "мокасины":   "moccasins: soft leather slip-on shoes with stitched seam on top, very flexible sole",
        "балетки":    "ballet flats: very flat thin sole, no heel at all, pointed or round toe, women's flat shoes",
        "угги":       "ugg boots: tall sheepskin boots, suede exterior, fluffy sheepskin lining visible at top",
    },
    "аксессуары": {
        "сумка":      "a handbag or shoulder bag: carried by handle or strap, structured or soft bag for women",
        "рюкзак":     "a backpack: two shoulder straps worn on the back, multiple compartments, school or travel bag",
        "клатч":      "a clutch: small flat evening bag held in the hand, no strap, formal occasion",
        "ремень":     "a belt: long narrow strip of leather or fabric with buckle, worn around the waist through pants loops",
        "шарф":       "a scarf: long rectangular fabric worn wrapped around the neck, winter or fashion accessory",
        "шапка":      "a beanie or knit hat: knitted fabric hat worn on the head, covers the ears, winter hat",
        "кепка":      "a baseball cap: structured cap with a stiff horizontal brim in front, adjustable strap at back",
        "панама":     "a bucket hat: soft hat with wide brim all the way around, floppy brim, summer hat",
        "перчатки":   "gloves: cover both hands separately, finger coverings, winter hand warmers",
        "носки":      "socks: short fabric tubes worn on the feet inside shoes, visible at the ankle",
        "украшения":  "jewelry: necklace, earrings, bracelet, ring, or other decorative body accessories",
        "очки":       "glasses or sunglasses: worn on the face with two lenses in frames over the eyes",
        "галстук":    "a necktie: long narrow strip of fabric knotted at the shirt collar, formal menswear",
        "часы":       "a wristwatch: timepiece worn on the wrist with a strap or bracelet, shows time",
    },
}

# --- Цвета (анализ пикселей) ---

COLOR_REFERENCES = {
    "белый":      (245, 245, 245),
    "чёрный":     (20,  20,  20),
    "серый":      (128, 128, 128),
    "синий":      (30,  80,  180),
    "красный":    (200, 30,  30),
    "зелёный":    (50,  150, 50),
    "бежевый":    (210, 190, 160),
    "коричневый": (120, 70,  40),
    "розовый":    (240, 150, 180),
    "жёлтый":     (240, 220, 30),
    "оранжевый":  (240, 130, 30),
    "бордовый":   (130, 20,  40),
    "фиолетовый": (130, 50,  180),
    "голубой":    (100, 180, 230),
}

# --- Стиль и сезон по категории ---

CATEGORY_TO_STYLE = {
    "футболка": "кэжуал", "лонгслив": "кэжуал", "поло": "кэжуал",
    "рубашка": "официальный", "блузка": "элегантный", "топ": "кэжуал",
    "боди": "элегантный", "свитер": "кэжуал", "худи": "уличный",
    "толстовка": "спортивный", "джемпер": "кэжуал", "кардиган": "кэжуал",
    "водолазка": "официальный", "жилет": "официальный", "пиджак": "официальный",
    "джинсы": "кэжуал", "брюки": "официальный", "чиносы": "кэжуал",
    "шорты": "кэжуал", "бермуды": "кэжуал", "юбка": "элегантный",
    "мини-юбка": "уличный", "леггинсы": "спортивный", "спортивные штаны": "спортивный",
    "пальто": "элегантный", "куртка": "уличный", "пуховик": "кэжуал",
    "тренч": "официальный", "плащ": "официальный", "ветровка": "спортивный",
    "бомбер": "уличный", "парка": "кэжуал", "дублёнка": "кэжуал",
    "платье": "элегантный", "вечернее платье": "элегантный", "сарафан": "кэжуал",
    "комбинезон": "уличный", "костюм": "официальный",
    "кроссовки": "спортивный", "туфли": "официальный", "ботинки": "уличный",
    "сапоги": "элегантный", "сандалии": "кэжуал", "шлёпанцы": "кэжуал",
    "лоферы": "официальный", "мокасины": "кэжуал", "балетки": "элегантный",
    "угги": "кэжуал", "сумка": "кэжуал", "рюкзак": "уличный",
    "клатч": "элегантный", "ремень": "официальный", "шарф": "кэжуал",
    "шапка": "кэжуал", "кепка": "уличный", "панама": "кэжуал",
    "перчатки": "официальный", "носки": "спортивный", "украшения": "элегантный",
    "очки": "кэжуал", "галстук": "официальный", "часы": "официальный",
}

CATEGORY_TO_SEASON = {
    "пуховик": "зима", "дублёнка": "зима", "парка": "зима",
    "пальто": "демисезон", "тренч": "демисезон", "плащ": "демисезон",
    "куртка": "демисезон", "ветровка": "демисезон", "бомбер": "демисезон",
    "шапка": "зима", "перчатки": "зима", "шарф": "зима", "угги": "зима",
    "сапоги": "зима", "ботинки": "демисезон",
    "сандалии": "лето", "шлёпанцы": "лето", "шорты": "лето",
    "бермуды": "лето", "сарафан": "лето", "панама": "лето",
}

SHOP_CATEGORIES = {
    "верх": ["футболка", "лонгслив", "поло", "рубашка", "блузка", "топ", "боди",
             "свитер", "худи", "толстовка", "джемпер", "кардиган", "водолазка",
             "жилет", "пиджак"],
    "низ": ["джинсы", "брюки", "чиносы", "шорты", "бермуды", "юбка",
            "мини-юбка", "леггинсы", "спортивные штаны"],
    "верхняя одежда": ["пальто", "куртка", "пуховик", "тренч", "плащ",
                       "ветровка", "бомбер", "парка", "дублёнка"],
    "платья и комбинезоны": ["платье", "вечернее платье", "сарафан", "комбинезон", "костюм"],
    "обувь": ["кроссовки", "туфли", "ботинки", "сапоги", "сандалии",
              "шлёпанцы", "лоферы", "мокасины", "балетки", "угги"],
    "аксессуары": ["сумка", "рюкзак", "клатч", "ремень", "шарф", "шапка",
                   "кепка", "панама", "перчатки", "носки", "украшения",
                   "очки", "галстук", "часы"],
}

MISSING_CATEGORIES_TIPS = {
    "верх": "Добавь базовую футболку или рубашку — они сделают образы более универсальными",
    "низ": "В гардеробе не хватает брюк или джинсов — основы большинства образов",
    "верхняя одежда": "Нет верхней одежды — пальто или куртка завершат любой образ",
    "платья и комбинезоны": "Платье или костюм добавят возможности для формальных образов",
    "обувь": "Без обуви образ не завершить — кроссовки или туфли must-have",
    "аксессуары": "Аксессуары (сумка, ремень) поднимают образ на новый уровень",
}


def _detect_color(image: Image.Image) -> str:
    w, h = image.size
    # берём центральные 60% изображения — там обычно сама вещь
    crop = image.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
    arr = np.array(crop.convert("RGB").resize((80, 80))).reshape(-1, 3).astype(float)

    # убираем белый и очень светлый фон
    mask = ~(np.all(arr > 215, axis=1))
    if mask.sum() > 30:
        arr = arr[mask]

    median = np.median(arr, axis=0)

    best_color = "чёрный"
    best_dist = float("inf")
    for name, ref in COLOR_REFERENCES.items():
        dist = float(np.sqrt(np.sum((median - np.array(ref)) ** 2)))
        if dist < best_dist:
            best_dist = dist
            best_color = name

    return best_color


def _clip_best_label(model, processor, image: Image.Image, prompts: list, labels: list) -> str:
    """Сравниваем батчами и берём глобально лучший логит."""
    BATCH = 20
    all_logits = []
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i:i + BATCH]
        inputs = processor(text=batch, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
        all_logits.extend(out.logits_per_image[0].tolist())

    best_idx = int(np.argmax(all_logits))
    # топ-5 для отладки
    top5 = sorted(enumerate(all_logits), key=lambda x: x[1], reverse=True)[:5]
    print("TOP-5:", [(labels[i], round(s, 3)) for i, s in top5])
    return labels[best_idx]


def _classify_sync(image_path: str) -> dict:
    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")

    # Проверяем — одежда или нет
    inputs = processor(text=CLOTHING_CHECK_LABELS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    if probs[1].item() > probs[0].item():
        return {"is_clothing": False}

    # Этап 1 — определяем широкую группу
    group_labels = list(GROUP_LABELS.keys())
    group_prompts = list(GROUP_LABELS.values())
    group = _clip_best_label(model, processor, image, group_prompts, group_labels)
    print(f"Группа: {group}")

    # Этап 2 — определяем конкретную категорию внутри группы
    subcat_dict = GROUP_CATEGORIES[group]
    cat_labels = list(subcat_dict.keys())
    cat_prompts = [f"a photo of {v}" for v in subcat_dict.values()]
    category = _clip_best_label(model, processor, image, cat_prompts, cat_labels)

    # Определяем цвет по центральному кропу
    color = _detect_color(image)

    style = CATEGORY_TO_STYLE.get(category, "кэжуал")
    season = CATEGORY_TO_SEASON.get(category, "всесезонный")
    description = f"{category.capitalize()} {color}го цвета в стиле {style}."

    return {
        "is_clothing": True,
        "category": category,
        "color": color,
        "style": style,
        "season": season,
        "description": description,
        "tags": [category, color, style],
    }


async def classify_image(image_path: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _classify_sync, image_path)


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
