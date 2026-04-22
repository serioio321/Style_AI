from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.database import ClothingItem, get_db
from app.services.outfit_generator import generate_outfits, analyze_outfit, describe_outfit

router = APIRouter(prefix="/outfits", tags=["outfits"])


@router.get("/generate")
async def get_outfits(db: Session = Depends(get_db)):
    items = db.query(ClothingItem).all()
    items_data = [
        {
            "id": item.id,
            "filename": item.filename,
            "category": item.category,
            "color": item.color,
            "style": item.style,
            "season": item.season,
        }
        for item in items
    ]

    if len(items_data) < 2:
        return {"outfits": [], "message": "Добавь минимум 2 вещи для генерации образов"}

    outfits = generate_outfits(items_data, max_outfits=3)

    for outfit in outfits:
        outfit["analysis"] = analyze_outfit(outfit)
        outfit["description"] = await describe_outfit(outfit)

    return {"outfits": outfits}
