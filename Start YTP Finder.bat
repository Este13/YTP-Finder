@echo off
title YTP Finder
cd /d "%~dp0"

curl -s -m 2 -o nul http://127.0.0.1:8765/api/status
if not errorlevel 1 (
    start "" http://127.0.0.1:8765
    exit /b
)

echo Starting YTP Finder... (keep this window open, close it to stop)
python app.py
if errorlevel 1 pause
