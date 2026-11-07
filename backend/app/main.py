"""FastAPI application entrypoint for NEXUS-ISP."""
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.model_loader import model_manager
from app.schemas.isp_schema import HealthResponse, ModelInfoResponse, ProcessResponse
from app.isp.pipeline import process_image_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nexus_isp.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing NEXUS-ISP backend...")
    try:
        model_manager.initialize()
        logger.info("NEXUS-ISP model and pipeline successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {e}")
    yield
    logger.info("Shutting down NEXUS-ISP backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for NEXUS-ISP: Hybrid Image Signal Processing using Classical ISP and Deep Learning",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        model_loaded=model_manager.model is not None,
        device=model_manager.device
    )

@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info():
    meta = model_manager.metadata
    desc = (
        "Hybrid ISP combines classical stages (Gray-world White Balance, "
        "bilinear Bayer demosaicing, Color Correction Matrix, sRGB gamma, and smoothstep tone curve) "
        "with a 63K-parameter residual depthwise-separable U-Net conditioned on MAD blind noise estimation."
    )
    return ModelInfoResponse(
        mode=meta["mode"],
        base=meta["base"],
        depth=meta["depth"],
        sep=meta["sep"],
        params=meta["params"],
        params_k=meta["params_k"],
        device=meta["device"],
        load_time_ms=meta["load_time_ms"],
        architecture_description=desc
    )

@app.post("/api/process", response_model=ProcessResponse)
async def process_image(
    file: UploadFile = File(...),
    noise_level: str = Form("medium"),
    custom_noise_a: float = Form(None),
    include_traditional: bool = Form(True)
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image format.")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        response = process_image_pipeline(
            image_bytes=contents,
            noise_preset=noise_level,
            custom_noise_a=custom_noise_a,
            include_traditional=include_traditional
        )
        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Error processing image in ISP pipeline")
        raise HTTPException(status_code=500, detail=f"ISP processing failure: {str(e)}")

# Mount frontend production build if available
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend build index.html not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
