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

## Project Structure

```
NEXUS-ISP/
├── .gitignore                   # Excludes checkpoints (*.pt), .venv, node_modules, dist
├── model/
│   └── hybrid_isp_final.pt      # Trained PyTorch checkpoint (preserved, untracked)
├── results/                     # Offline benchmark CSVs and comparison grid
│   ├── quality.csv
│   ├── efficiency.csv
│   └── comparison.png
├── source/                      # Original training and evaluation scripts
│   ├── isp_lib.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── backend/                     # FastAPI Backend Server
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py        # Settings and CORS
│   │   │   └── model_loader.py  # Model manager & warm-up
│   │   ├── isp/
│   │   │   ├── isp_lib.py       # Classical ISP & RAW simulation
│   │   │   ├── models.py        # Compact U-Net PyTorch definition
│   │   │   └── pipeline.py      # End-to-end processing & Base64 encoding
│   │   ├── schemas/
│   │   │   └── isp_schema.py    # Request/Response Pydantic schemas
│   │   └── main.py              # FastAPI endpoints & static file serving
│   ├── tests/
│   │   ├── test_model_inference.py  # Standalone PyTorch forward-pass test
│   │   └── test_api_endpoints.py    # End-to-end FastAPI integration test
│   └── requirements.txt
├── frontend/                    # Modern React + Vite Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── NoiseControls.jsx
│   │   │   ├── ComparisonViewer.jsx
│   │   │   ├── MetricsDashboard.jsx
│   │   │   ├── DownloadBar.jsx
│   │   │   └── ArchitectureModal.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── run_backend.bat              # 1-click script to start FastAPI backend
├── run_frontend.bat             # 1-click script to start Vite dev server
└── README.md
```

---

## Getting Started

### 1. Backend Setup & Testing

Activate the virtual environment and run the test suites:

```powershell
# Run standalone model inference test
.\.venv\Scripts\python.exe backend\tests\test_model_inference.py

# Run FastAPI API endpoint integration test suite
.\.venv\Scripts\python.exe -u backend\tests\test_api_endpoints.py
```

To run the backend server:

```powershell
.\run_backend.bat
# Or manually:
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`. Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup

In a separate terminal:

```powershell
.\run_frontend.bat
# Or manually:
cd frontend
npm run dev
```

Open your browser to `http://localhost:5173`.

---

## API Endpoints

- `GET /api/health` – Returns health status, model load status, and device (CPU/CUDA).
- `GET /api/model-info` – Returns architecture details, parameter count (63,347), and checkpoint details.
- `POST /api/process` – Multipart form upload accepting `file`, `noise_level` (`low`, `medium`, `high`, `custom`), `custom_noise_a`, and `include_traditional`. Returns Base64 images and full benchmark metrics.
