from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///./wardrobe.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ClothingItem(Base):
    __tablename__ = "clothing_items"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    category = Column(String)        # футболка, джинсы, платье, ...
    color = Column(String)           # белый, чёрный, синий, ...
    style = Column(String)           # casual, formal, sport, ...
    season = Column(String)          # лето, зима, демисезон
    description = Column(Text)       # полное описание от GPT
    embedding = Column(Text)         # JSON-список float (CLIP-like от GPT)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
