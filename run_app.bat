@echo off
echo Starting Chest X-Ray Disease Detection System...
echo.

REM Start FastAPI Backend
start "Backend (FastAPI)" cmd /k ".\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

REM Start Vite Frontend
cd frontend
start "Frontend (Vite)" cmd /k "npm run dev"

echo Backend started on http://127.0.0.1:8000
echo Frontend started on http://127.0.0.1:5173
echo.
echo Application is opening in your browser...
timeout /t 3 >nul
start http://127.0.0.1:5173
