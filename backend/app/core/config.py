"""Application configuration and settings."""
import os
from pathlib import Path
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "hybrid_isp_final.pt"

class Settings(BaseModel):
    PROJECT_NAME: str = "NEXUS-ISP API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "1536"))  # Max dimension for real-time responsiveness
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]

settings = Settings()
