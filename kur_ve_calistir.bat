@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [Sukut] Ilk kurulum yapiliyor, birkac dakika surebilir...
  py -3.12 -m venv .venv || (echo Python 3.12 gerekli: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\pip install -q wheels\DeepFilterLib-0.5.6-cp312-none-win_amd64.whl
  .venv\Scripts\pip install -q -r requirements.txt
)
start "" .venv\Scripts\pythonw.exe sukut.py
