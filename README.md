# NEXUS-ISP: Hybrid Image Signal Processing

A modern web application and production-ready deep learning system combining **classical camera ISP color science** with a **compact residual deep learning U-Net** (63.35K parameters, ~0.24 MB) trained on synthetic RAW unprocessing.

---

## Key Features

1. **True Hybrid Architecture**:
   - **Classical Front-End**: Gray-world White Balance, Bilinear RGGB Demosaicing, Color Correction Matrix (CCM), sRGB linear conversion, and Smoothstep Tone Curve.
   - **Neural Residual Correction**: Depthwise-separable 3-stage U-Net conditioned on Median Absolute Deviation (MAD) blind noise estimation $\sigma$. Learns purely the residual correction $\Delta_{\text{correction}}$ so it remains compact (63,347 params) and color-stable.
2. **Real Model Inference (No Mocks)**:
   - Direct PyTorch inference using the trained `model/hybrid_isp_final.pt` checkpoint.
   - Preserves exact model architecture and weight parameters.
3. **Rigorous Synthetic RAW Handling**:
   - Accurately inverts compressed sRGB inputs via Brooks et al. (CVPR 2019) *unprocessing* into Bayer RGGB sensor space with heteroscedastic Poisson-Gaussian noise ($\sigma^2 = a \cdot x + b$).
4. **Comprehensive Benchmarking**:
   - Real-time latency measurements (Classical latency, Neural latency, Total latency).
   - Quantitative evaluation metrics (PSNR and SSIM against ground truth and naive baseline).
5. **Interactive UI**:
   - Draggable split-slider before/after viewer.
   - 4-way side-by-side comparison grid (Ground Truth, Naive Noisy Front-End, Traditional OpenCV ISP, Hybrid Neural ISP).
   - Built-in test samples for 1-click evaluation.
   - Noise level controls (Low, Medium, High, Custom slider).
   - Direct image downloads and metrics clipboard copying.

---

## Deployment on Render

This project includes a production multi-stage `Dockerfile` and `render.yaml` Blueprint.

### 1. Push to GitHub
```bash
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/NEXUS-ISP.git
git branch -M main
git push -u origin main
```

### 2. Deploy on Render
1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** &rarr; **Web Service**.
3. Connect your GitHub repository `NEXUS-ISP`.
4. Choose **Docker** as the environment.
5. In **Environment Variables**, add:
   - `MODEL_BASE64`: Copy and paste the contents of `model_base64.txt` (generated locally in your project root, excluded from Git). This allows Render to instantly restore your checkpoint at startup without committing weights to Git!
6. Click **Create Web Service**.

Render will automatically build the React frontend, start the FastAPI server, bind to `$PORT`, and provide your live HTTPS URL.

---

## Local Development & Testing

### 1. Backend Setup & Testing

```powershell
# Run standalone model inference test
.\.venv\Scripts\python.exe backend\tests\test_model_inference.py

# Run FastAPI API endpoint integration test suite
.\.venv\Scripts\python.exe -u backend\tests\test_api_endpoints.py
```

To run the backend server locally:

```powershell
.\run_backend.bat
# Or manually:
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`. Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Development

```powershell
.\run_frontend.bat
# Or manually:
cd frontend
npm run dev
```

Open your browser to `http://localhost:5173`.
