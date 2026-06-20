@echo off
setlocal
cd /d "%~dp0\..\.."

if not exist .venv (
  py -3 -m venv .venv
)

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r app\requirements.txt
.venv\Scripts\python.exe app\run_web.py
