"""Standalone test script to verify model loading, architecture, and inference on hybrid_isp_final.pt."""
import os
import sys
import time
import numpy as np
import torch

# Ensure backend/app is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.isp.models import load_ckpt, count_params, HybridISP
import app.isp.isp_lib as L

def test_hybrid_isp_inference():
    ckpt_path = os.path.join(os.path.dirname(backend_dir), "model", "hybrid_isp_final.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    print(f"[TEST] Found checkpoint at: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TEST] Using device: {device}")

    # 1. Test Model Loading
    t0 = time.perf_counter()
    model, ck = load_ckpt(ckpt_path, device=device)
    load_time_ms = (time.perf_counter() - t0) * 1e3
    print(f"[TEST] Model loaded in {load_time_ms:.2f} ms")
    print(f"[TEST] Checkpoint metadata: mode={ck.get('mode')}, base={ck.get('base')}, depth={ck.get('depth')}, sep={ck.get('sep')}")

    assert isinstance(model, HybridISP), "Loaded model is not an instance of HybridISP!"
    params = count_params(model)
    print(f"[TEST] Total parameters: {params:,} ({params/1e3:.2f} K)")
    assert 60000 < params < 70000, f"Unexpected parameter count: {params}"

    # 2. Test Input Processing
    H, W = 256, 256
    # Create synthetic RGB test image with gradients and structures
    x = np.linspace(0, 1, W, dtype=np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    test_img = np.stack([xx, yy, 0.5 * (xx + yy)], axis=-1)

    rng = np.random.default_rng(42)
    noise_a = 2e-3  # mid-noise condition
    print(f"[TEST] Simulating RAW with noise_a={noise_a}...")
    raw, meta = L.simulate_raw(test_img, rng=rng, noise_a=noise_a)
    assert raw.shape == (H, W), f"Raw shape mismatch: {raw.shape}"

    sigma = L.estimate_raw_sigma(raw)
    print(f"[TEST] Estimated raw sigma: {sigma:.5f}")

    # Classical front-end
    t_front0 = time.perf_counter()
    front = L.isp_front(raw)
    front_time_ms = (time.perf_counter() - t_front0) * 1e3
    print(f"[TEST] Classical front-end executed in {front_time_ms:.2f} ms, shape: {front.shape}")

    # 3. Model Forward Pass
    s_tensor = torch.tensor([sigma], dtype=torch.float32, device=device)
    x_tensor = torch.from_numpy(front.transpose(2, 0, 1)[None]).float().to(device)

    # Warm-up pass
    with torch.no_grad():
        _ = model(x_tensor, s_tensor)

    # Timed inference passes
    runs = 10
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t_inf0 = time.perf_counter()
            out_tensor = model(x_tensor, s_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t_inf0) * 1e3)

    median_lat = np.median(latencies)
    print(f"[TEST] Inference latency (median of {runs} runs): {median_lat:.2f} ms")

    enhanced = out_tensor.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    assert enhanced.shape == (H, W, 3), f"Output shape mismatch: {enhanced.shape}"
    assert not np.isnan(enhanced).any(), "Output contains NaN values!"
    assert not np.isinf(enhanced).any(), "Output contains Inf values!"
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0, "Output values out of [0, 1] bounds!"

    # Traditional classical ISP comparison
    trad = L.isp_traditional(raw)

    # 4. Check MSE / PSNR
    mse_front = np.mean((front - test_img) ** 2)
    mse_hybrid = np.mean((enhanced - test_img) ** 2)
    mse_trad = np.mean((trad - test_img) ** 2)
    psnr_front = -10 * np.log10(mse_front + 1e-10)
    psnr_hybrid = -10 * np.log10(mse_hybrid + 1e-10)
    psnr_trad = -10 * np.log10(mse_trad + 1e-10)

    print(f"[TEST] Naive PSNR:       {psnr_front:.2f} dB")
    print(f"[TEST] Traditional PSNR: {psnr_trad:.2f} dB")
    print(f"[TEST] Hybrid ISP PSNR:  {psnr_hybrid:.2f} dB")
    print("[SUCCESS] All model loading, preprocessing, inference, and validity tests passed!")

if __name__ == "__main__":
    test_hybrid_isp_inference()
