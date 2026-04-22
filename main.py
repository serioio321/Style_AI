from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.models.database import init_db
from app.routers import wardrobe, outfits

app = FastAPI(title="StyleAI — персональный стилист")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(wardrobe.router)
app.include_router(outfits.router)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
