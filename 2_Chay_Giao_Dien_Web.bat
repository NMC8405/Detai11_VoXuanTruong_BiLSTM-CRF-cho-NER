@echo off
chcp 65001 > nul
title NER BiLSTM-CRF — Giao Dien Web

echo.
echo ============================================================
echo   NER BiLSTM-CRF  --  Giao Dien Web
echo ============================================================
echo.

:: Kiem tra Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python!
    pause
    exit /b 1
)

:: Kiem tra uvicorn
python -c "import uvicorn" > nul 2>&1
if errorlevel 1 (
    echo Dang cai uvicorn...
    pip install uvicorn[standard] fastapi --quiet
)

echo Khoi dong server tai: http://localhost:8000
echo.
echo Giu Ctrl+C de dung server.
echo.

:: Mo trinh duyet sau 2 giay
start /min cmd /c "timeout /t 2 > nul && start http://localhost:8000"

:: Chay server
python -m uvicorn web_app.app:app --host 0.0.0.0 --port 8000 --reload

pause
