"""Pydantic schemas for request and response validation."""
from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class NoisePreset(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CUSTOM = "custom"

# Mapping preset names to synthetic noise variance coefficient 'a'
NOISE_PRESET_MAP = {
    NoisePreset.LOW: 2e-4,     # Good lighting, minimal noise
    NoisePreset.MEDIUM: 2e-3,  # Typical indoor / moderate noise
    NoisePreset.HIGH: 1e-2     # Low-light / high sensor gain
}

class ModelInfoResponse(BaseModel):
    mode: str
    base: int
    depth: int
    sep: bool
    params: int
    params_k: float
    device: str
    load_time_ms: float
    architecture_description: str

class ProcessMetrics(BaseModel):
    estimated_sigma: float
    noise_preset: str
    noise_a: float
    classical_latency_ms: float
    neural_latency_ms: float
    total_latency_ms: float
    image_shape: List[int]
    psnr_hybrid_vs_groundtruth: Optional[float] = None
    ssim_hybrid_vs_groundtruth: Optional[float] = None
    psnr_naive_vs_groundtruth: Optional[float] = None
    ssim_naive_vs_groundtruth: Optional[float] = None
    psnr_traditional_vs_groundtruth: Optional[float] = None
    ssim_traditional_vs_groundtruth: Optional[float] = None

class ProcessResponse(BaseModel):
    success: bool
    metrics: ProcessMetrics
    original_b64: str           # The ground truth RGB (Base64 JPEG)
    naive_b64: str              # Classical front-end without denoising (Base64 JPEG)
    traditional_b64: Optional[str] = None # Classical ISP (NLM + unsharp) (Base64 JPEG)
    enhanced_b64: str           # NEXUS-ISP Hybrid enhanced output (Base64 JPEG)
    explanation: str

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    device: str
