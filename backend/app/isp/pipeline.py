"""End-to-end execution pipeline for NEXUS-ISP."""
import base64
import time
import io
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from app.core.config import settings
from app.core.model_loader import model_manager
import app.isp.isp_lib as L
from app.schemas.isp_schema import (
    NoisePreset,
    NOISE_PRESET_MAP,
    ProcessMetrics,
    ProcessResponse
)

def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into an RGB float32 array in range [0, 1]."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode the uploaded image file.")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Resize if image exceeds max dimension
    h, w = rgb.shape[:2]
    max_dim = settings.MAX_IMAGE_SIZE
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # U-Net with 3 pooling stages requires dimensions divisible by 16
    h_adj = (rgb.shape[0] // 16) * 16
    w_adj = (rgb.shape[1] // 16) * 16
    if h_adj < 16 or w_adj < 16:
        raise ValueError("Image dimensions are too small (must be at least 16x16).")

    return np.ascontiguousarray(rgb[:h_adj, :w_adj])


def encode_to_base64_jpeg(rgb_float: np.ndarray, quality: int = 92) -> str:
    """Encode an RGB float32 image into a base64 Data URL string."""
    rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("Failed to encode image to JPEG.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def process_image_pipeline(
    image_bytes: bytes,
    noise_preset: str = "medium",
    custom_noise_a: float = None,
    include_traditional: bool = True
) -> ProcessResponse:
    """Execute the complete Hybrid ISP pipeline and comparative baselines."""
    # 1. Decode & normalize
    ground_truth = decode_image(image_bytes)
    h, w, _ = ground_truth.shape

    # 2. Determine noise parameter
    preset_enum = NoisePreset(noise_preset.lower()) if noise_preset.lower() in [e.value for e in NoisePreset] else NoisePreset.MEDIUM
    if preset_enum == NoisePreset.CUSTOM and custom_noise_a is not None:
        noise_a = float(np.clip(custom_noise_a, 1e-5, 5e-2))
    else:
        noise_a = NOISE_PRESET_MAP.get(preset_enum, 2e-3)

    # 3. Simulate camera RAW capture (unprocessing: inverse tone -> linear -> inverse CCM -> inverse WB -> mosaic + noise)
    rng = np.random.default_rng(100)
    raw, meta = L.simulate_raw(ground_truth, rng=rng, noise_a=noise_a)

    # 4. Blind noise estimation
    estimated_sigma = L.estimate_raw_sigma(raw)

    # 5. Classical front-end (WB -> demosaic -> CCM -> gamma -> tone)
    t_classical_0 = time.perf_counter()
    front = L.isp_front(raw)
    classical_latency_ms = (time.perf_counter() - t_classical_0) * 1e3

    # 6. Hybrid ISP Neural residual inference
    model = model_manager.model
    device = model_manager.device

    t_neural_0 = time.perf_counter()
    s_tensor = torch.tensor([estimated_sigma], dtype=torch.float32, device=device)
    front_tensor = torch.from_numpy(front.transpose(2, 0, 1)[None]).float().to(device)

    with torch.no_grad():
        out_tensor = model(front_tensor, s_tensor)
        if device == "cuda":
            torch.cuda.synchronize()

    neural_latency_ms = (time.perf_counter() - t_neural_0) * 1e3
    total_latency_ms = classical_latency_ms + neural_latency_ms

    hybrid_enhanced = out_tensor.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    # 7. Optional Traditional Classical ISP (NLM + unsharp)
    trad_output = None
    trad_b64 = None
    psnr_trad = None
    ssim_trad = None
    if include_traditional:
        trad_output = L.isp_traditional(raw)
        trad_b64 = encode_to_base64_jpeg(trad_output)
        psnr_trad = round(float(psnr_fn(ground_truth, trad_output, data_range=1.0)), 2)
        ssim_trad = round(float(ssim_fn(ground_truth, trad_output, channel_axis=2, data_range=1.0)), 4)

    # 8. Compute quantitative metrics
    psnr_hybrid = round(float(psnr_fn(ground_truth, hybrid_enhanced, data_range=1.0)), 2)
    ssim_hybrid = round(float(ssim_fn(ground_truth, hybrid_enhanced, channel_axis=2, data_range=1.0)), 4)

    psnr_naive = round(float(psnr_fn(ground_truth, front, data_range=1.0)), 2)
    ssim_naive = round(float(ssim_fn(ground_truth, front, channel_axis=2, data_range=1.0)), 4)

    # 9. Encode images to Base64 for instant display
    original_b64 = encode_to_base64_jpeg(ground_truth)
    naive_b64 = encode_to_base64_jpeg(front)
    enhanced_b64 = encode_to_base64_jpeg(hybrid_enhanced)

    metrics = ProcessMetrics(
        estimated_sigma=round(estimated_sigma, 5),
        noise_preset=preset_enum.value,
        noise_a=noise_a,
        classical_latency_ms=round(classical_latency_ms, 2),
        neural_latency_ms=round(neural_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        image_shape=[h, w, 3],
        psnr_hybrid_vs_groundtruth=psnr_hybrid,
        ssim_hybrid_vs_groundtruth=ssim_hybrid,
        psnr_naive_vs_groundtruth=psnr_naive,
        ssim_naive_vs_groundtruth=ssim_naive,
        psnr_traditional_vs_groundtruth=psnr_trad,
        ssim_traditional_vs_groundtruth=ssim_trad
    )

    explanation = (
        f"Input converted to Bayer RAW with noise level a={noise_a:.1e}. "
        f"Classical front-end restored color & tone in {classical_latency_ms:.1f}ms. "
        f"Residual U-Net performed deep denoising in {neural_latency_ms:.1f}ms "
        f"yielding PSNR improvement from {psnr_naive} dB (naive) to {psnr_hybrid} dB (hybrid)."
    )

    return ProcessResponse(
        success=True,
        metrics=metrics,
        original_b64=original_b64,
        naive_b64=naive_b64,
        traditional_b64=trad_b64,
        enhanced_b64=enhanced_b64,
        explanation=explanation
    )
