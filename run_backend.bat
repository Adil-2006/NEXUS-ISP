@echo off
echo ===================================================
echo Starting NEXUS-ISP Backend Server (FastAPI + PyTorch)
echo ===================================================
call .venv\Scripts\activate
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
