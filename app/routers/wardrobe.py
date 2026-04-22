import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from app.models.database import ClothingItem, get_db
from app.services.classifier import classify_image, analyze_gaps, CATEGORY_TO_STYLE, CATEGORY_TO_SEASON

router = APIRouter(prefix="/wardrobe", tags=["wardrobe"])

UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/upload")
async def upload_item(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Допустимые форматы: JPG, PNG, WEBP")

    filename = f"{uuid.uuid4()}{suffix}"
    save_path = UPLOAD_DIR / filename

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        classification = await classify_image(str(save_path))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {str(e)}")

    if not classification.get("is_clothing", True):
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="На фото не обнаружена одежда, обувь или аксессуар")

    item = ClothingItem(
        filename=filename,
        category=classification.get("category", ""),
        color=classification.get("color", ""),
        style=classification.get("style", ""),
        season=classification.get("season", ""),
        description=classification.get("description", ""),
        embedding=str(classification.get("tags", [])),
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    return {
        "id": item.id,
        "filename": item.filename,
        "category": item.category,
        "color": item.color,
        "style": item.style,
        "season": item.season,
        "description": item.description,
        "tags": classification.get("tags", []),
    }


@router.get("/items")
def get_items(db: Session = Depends(get_db)):
    items = db.query(ClothingItem).order_by(ClothingItem.created_at.desc()).all()
    return [
        {
            "id": item.id,
            "filename": item.filename,
            "category": item.category,
            "color": item.color,
            "style": item.style,
            "season": item.season,
            "description": item.description,
        }
        for item in items
    ]


class ItemUpdate(BaseModel):
    category: Optional[str] = None
    color: Optional[str] = None


@router.patch("/items/{item_id}")
def update_item(item_id: int, data: ItemUpdate, db: Session = Depends(get_db)):
    item = db.query(ClothingItem).filter(ClothingItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Вещь не найдена")
    if data.category is not None:
        item.category = data.category
        item.style = CATEGORY_TO_STYLE.get(data.category, item.style)
        item.season = CATEGORY_TO_SEASON.get(data.category, item.season)
    if data.color is not None:
        item.color = data.color
    db.commit()
    db.refresh(item)
    return {"id": item.id, "category": item.category, "color": item.color, "style": item.style, "season": item.season}


@router.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ClothingItem).filter(ClothingItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Вещь не найдена")

    path = UPLOAD_DIR / item.filename
    path.unlink(missing_ok=True)

    db.delete(item)
    db.commit()
    return {"ok": True}


@router.get("/gaps")
def get_gaps(db: Session = Depends(get_db)):
    items = db.query(ClothingItem).all()
    items_data = [{"category": i.category, "color": i.color, "style": i.style} for i in items]
    tips = analyze_gaps(items_data)
    return {"tips": tips}
