from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from routers import centers, health, order_entry
from utils.logger import setup_logging

setup_logging()
settings = get_settings()

app = FastAPI(title="Uruno OCR Backend", version=settings.app_version)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(centers.router)
app.include_router(order_entry.router)
