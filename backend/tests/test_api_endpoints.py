"""Integration test suite for NEXUS-ISP FastAPI endpoints.
Uses httpx.AsyncClient and httpx.ASGITransport directly for fast, reliable, asynchronous API testing without AnyIO thread portal issues.
"""
import asyncio
import os
import sys
import cv2
import numpy as np
import httpx

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.core.model_loader import model_manager

async def run_integration_tests():
    print("=" * 60)
    print("NEXUS-ISP FastAPI Integration Test Suite")
    print("=" * 60)

    # Explicitly ensure model is initialized
    model_manager.initialize()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Test 1: Health Check
        print("\n[TEST 1] GET /api/health ...")
        res = await client.get("/api/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        health = res.json()
        print(f"  Status: {health['status']}, Model Loaded: {health['model_loaded']}, Device: {health['device']}")
        assert health["status"] == "ok"
        assert health["model_loaded"] is True
        print("  -> PASSED")

        # Test 2: Model Information
        print("\n[TEST 2] GET /api/model-info ...")
        res = await client.get("/api/model-info")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        info = res.json()
        print(f"  Mode: {info['mode']}, Params: {info['params']:,} ({info['params_k']}K), Base: {info['base']}, Depth: {info['depth']}")
        assert info["mode"] == "hybrid"
        assert info["params"] == 63347
        print("  -> PASSED")

        # Test 3: Process Image Pipeline (Medium Noise)
        print("\n[TEST 3] POST /api/process (Medium Noise) ...")
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[:, :] = (120, 80, 200)
        cv2.circle(img, (64, 64), 36, (255, 255, 255), -1)
        cv2.putText(img, "NEXUS", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        success, encoded = cv2.imencode(".png", img)
        assert success, "Image encoding failed"

        files = {"file": ("test_input.png", encoded.tobytes(), "image/png")}
        data = {"noise_level": "medium", "include_traditional": "true"}

        res = await client.post("/api/process", files=files, data=data)
        assert res.status_code == 200, f"Process request failed: {res.text}"
        result = res.json()
        assert result["success"] is True
        m = result["metrics"]
        print(f"  Classical Latency: {m['classical_latency_ms']:.2f} ms")
        print(f"  Neural Latency:    {m['neural_latency_ms']:.2f} ms")
        print(f"  Total Latency:     {m['total_latency_ms']:.2f} ms")
        print(f"  Estimated Sigma:   {m['estimated_sigma']:.5f}")
        print(f"  Naive PSNR:        {m['psnr_naive_vs_groundtruth']} dB")
        print(f"  Traditional PSNR:  {m['psnr_traditional_vs_groundtruth']} dB")
        print(f"  Hybrid ISP PSNR:   {m['psnr_hybrid_vs_groundtruth']} dB")
        print(f"  Explanation:       {result['explanation']}")

        assert result["enhanced_b64"].startswith("data:image/jpeg;base64,")
        assert result["naive_b64"].startswith("data:image/jpeg;base64,")
        assert result["original_b64"].startswith("data:image/jpeg;base64,")
        print("  -> PASSED")

        # Test 4: Process with High Noise
        print("\n[TEST 4] POST /api/process (High Noise / Low Light) ...")
        files_high = {"file": ("test_high.png", encoded.tobytes(), "image/png")}
        data_high = {"noise_level": "high", "include_traditional": "false"}
        res_high = await client.post("/api/process", files=files_high, data=data_high)
        assert res_high.status_code == 200
        result_high = res_high.json()
        assert result_high["success"] is True
        m_high = result_high["metrics"]
        print(f"  High Noise Sigma:  {m_high['estimated_sigma']:.5f}")
        print(f"  Hybrid PSNR:       {m_high['psnr_hybrid_vs_groundtruth']} dB")
        print("  -> PASSED")

    print("\n" + "=" * 60)
    print("ALL 4 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_integration_tests())
