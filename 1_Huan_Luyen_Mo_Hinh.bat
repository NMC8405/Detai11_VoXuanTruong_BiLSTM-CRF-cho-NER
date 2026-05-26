@echo off
chcp 65001 > nul
title NER BiLSTM-CRF — Huan Luyen Mo Hinh

echo.
echo ============================================================
echo   NER BiLSTM-CRF  --  Huan Luyen Mo Hinh
echo ============================================================
echo.

:: Kiem tra Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python! Hay cai Python 3.9+ truoc.
    pause
    exit /b 1
)

:: Cai thu vien neu chua co
echo [1/3] Kiem tra thu vien...
pip install -r requirements.txt --quiet

:: Chuan bi du lieu
echo.
echo [2/3] Chuan bi du lieu...
if not exist "src\data\eng.train" (
    python prepare_data.py
    if errorlevel 1 (
        echo [LOI] Khong the chuan bi du lieu!
        pause
        exit /b 1
    )
) else (
    echo      Du lieu da san sang.
)

:: Huan luyen
echo.
echo [3/3] Bat dau huan luyen...
echo      (Co the mat 5-30 phut tuy thiet bi)
echo.
python run_training.py --epochs 5

if errorlevel 1 (
    echo.
    echo [LOI] Training that bai. Xem log o tren.
) else (
    echo.
    echo [4/4] Dang thuc hien phan tich loi va danh gia model...
    python error_analysis.py
    echo.
    echo ============================================================
    echo   Huan luyen va danh gia hoan tat!
    echo   Chay file 2_Chay_Giao_Dien_Web.bat de xem ket qua.
    echo ============================================================
)

echo.
pause
